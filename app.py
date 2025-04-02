import os
import tempfile
import sounddevice as sd
import soundfile as sf
import numpy as np
import threading
import queue
import google.generativeai as genai
from gtts import gTTS
from playsound import playsound
import speech_recognition as sr
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up your API key - You'll need to get this from Google AI Studio
API_KEY = os.getenv("GEMINI_API_KEY")  # Load from environment variable
if not API_KEY:
    raise ValueError("Please set GEMINI_API_KEY in .env file")
genai.configure(api_key=API_KEY)

# Configure the model
generation_config = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 1024,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

# Initialize the model - Updated to use the correct model name
try:
    # Try the newer model naming convention
    model = genai.GenerativeModel(model_name="gemini-2.0-flash-exp",
                                generation_config=generation_config,
                                safety_settings=safety_settings)
    conversation = model.start_chat(history=[])
    print("Connected to Gemini using model")
except Exception:
    try:
        # Fallback to the newest model if available
        model = genai.GenerativeModel(model_name="gemini-1.5-pro",
                                    generation_config=generation_config,
                                    safety_settings=safety_settings)
        conversation = model.start_chat(history=[])
        print("Connected to Gemini using model: gemini-1.5-pro")
    except Exception as e:
        print(f"Failed to initialize Gemini model: {e}")
        print("Attempting to list available models...")
        try:
            for m in genai.list_models():
                if "gemini" in m.name.lower():
                    print(f"Available Gemini model: {m.name}")
            print("Please update the script with one of these model names.")
        except:
            print("Unable to list models. Please check your API key and internet connection.")
        exit(1)

class AudioChatbot:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.recording_thread = None
        
    def record_audio(self):
        """Record audio from the microphone and add to queue."""
        self.is_recording = True
        print("Listening... Press Ctrl+C to stop.")
        
        try:
            while self.is_recording:
                with sr.Microphone() as source:
                    print("Say something...")
                    self.recognizer.adjust_for_ambient_noise(source)
                    audio = self.recognizer.listen(source)
                    self.audio_queue.put(audio)
                    print("Processing speech...")
        except KeyboardInterrupt:
            self.is_recording = False
        except Exception as e:
            print(f"Error recording audio: {e}")
            self.is_recording = False
            
    def start_recording(self):
        """Start recording in a separate thread."""
        if self.recording_thread is None or not self.recording_thread.is_alive():
            self.is_recording = True
            self.recording_thread = threading.Thread(target=self.record_audio)
            self.recording_thread.daemon = True
            self.recording_thread.start()
            
    def stop_recording(self):
        """Stop the recording thread."""
        self.is_recording = False
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=1)
    
    def process_audio(self):
        """Process audio from the queue and get responses from Gemini."""
        try:
            while True:
                # Get audio from queue (non-blocking with timeout)
                try:
                    audio = self.audio_queue.get(timeout=0.5)
                except queue.Empty:
                    if not self.is_recording:
                        break
                    continue
                
                try:
                    # Convert speech to text
                    text = self.recognizer.recognize_google(audio)  # Using Google's free speech recognition
                    print(f"You said: {text}")
                    
                    # If the user says "exit" or "quit", stop the chatbot
                    if text.lower() in ["exit", "quit", "stop"]:
                        print("Exiting chatbot...")
                        self.is_recording = False
                        break
                    
                    # Get response from Gemini
                    response = conversation.send_message(text)
                    response_text = response.text
                    print(f"Gemini: {response_text}")
                    
                    # Convert text to speech
                    self.text_to_speech(response_text)
                    
                except sr.UnknownValueError:
                    print("Sorry, I couldn't understand what you said.")
                except sr.RequestError as e:
                    print(f"Could not request results from Google Speech Recognition service; {e}")
                except Exception as e:
                    print(f"Error processing audio: {e}")
                
                self.audio_queue.task_done()
        except KeyboardInterrupt:
            print("Processing stopped by user.")
        
    def text_to_speech(self, text):
        """Convert text to speech and play it."""
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                temp_filename = temp_file.name
            
            # Generate speech
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(temp_filename)
            
            # Play speech
            playsound(temp_filename)
            
            # Clean up
            os.unlink(temp_filename)
        except Exception as e:
            print(f"Error in text-to-speech: {e}")
    
    def run(self):
        """Run the chatbot."""
        print("Starting Audio Chatbot with Gemini...")
        print("You can speak to the chatbot. Say 'exit', 'quit', or 'stop' to end the conversation.")
        
        try:
            # Start recording in a separate thread
            self.start_recording()
            
            # Process audio in the main thread
            self.process_audio()
        except KeyboardInterrupt:
            print("\nStopping the chatbot...")
        finally:
            self.stop_recording()
            print("Chatbot stopped.")

if __name__ == "__main__":
    chatbot = AudioChatbot()
    chatbot.run()