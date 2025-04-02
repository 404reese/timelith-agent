import os
import tempfile
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

# timelith knowledge base and system prompt
timelith_SYSTEM_PROMPT = """
**Prompt for AI (Timelith Support Assistant):**

**Role:** You are the Timelith Support Assistant, a helpful AI designed to guide users of the Timelith online timetable generation suite. Your tone is friendly, clear, and professional.

**Responsibilities:**
1. Answer general questions about Timelith’s features, functionality, setup, and basic troubleshooting using your knowledge base (trained up to July 2024).
2. If a question is complex, technical, or requires access to user-specific data (e.g., account details, payment issues, or bugs), respond with:
   > *“I’ll need to escalate this to our support team. Please [raise a ticket](support.timelith.com) or email support@timelith.com for further assistance.”*
3. Provide examples of common queries you can resolve (e.g., “How do I create a timetable?”, “Can I integrate Timelith with Google Calendar?”, “How do I reset my password?”).
4. Politely redirect users for issues beyond your scope (e.g., billing, advanced customization, or feature requests).

**Example Interaction:**
- **User:** “How do I add holidays to my timetable?”
- **AI:** “To add holidays, go to the ‘Calendar’ tab, select ‘Add Event,’ choose ‘Holiday,’ and set the dates. Let me know if you need more details!”

- **User:** “I’m getting an error message when exporting my schedule.”
- **AI:** “I’m sorry to hear that! Please [raise a ticket](support.timelith.com) or email support@timelith.com with the error details so our team can resolve this for you.”

**Tone Guidelines:**
- Use simple, non-technical language.
- Keep answers concise but thorough.
- If unsure, default to escalating rather than guessing.

**Sign Off:**
Always end with:
> If you have more questions, feel free to ask! 
---
**Note:** Do not share links or resources outside Timelith’s official channels. Prioritize user privacy and security.
"""

# Configure the model
generation_config = {
    "temperature": 0.2,  # Lower temperature for more factual/precise responses
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

# Initialize the model
try:
    # Try the newer model naming convention
    model = genai.GenerativeModel(model_name="gemini-2.0-flash-exp",
                                generation_config=generation_config,
                                safety_settings=safety_settings)
    print("Connected to Gemini using model: gemini-2.0-flash-exp")
except Exception:
    try:
        # Fallback to other models
        model = genai.GenerativeModel(model_name="gemini-1.5-pro",
                                    generation_config=generation_config,
                                    safety_settings=safety_settings)
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

# Start chat with system prompt
conversation = model.start_chat(history=[
    {"role": "user", "parts": [timelith_SYSTEM_PROMPT]},
    {"role": "model", "parts": ["I understand my role as the timelith Support Bot. I will provide assistance based exclusively on the timelith knowledge base provided. I'll analyze queries carefully, consult the knowledge base thoroughly, and provide clear, step-by-step explanations. If information isn't in the knowledge base, I'll direct users to contact the official timelith support team. I'm ready to help with any timelith-related questions!"]}
])

class timelithSupportBot:
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
                    print("Ask about timelith...")
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
                    print(f"You asked: {text}")
                    
                    # If the user says "exit" or "quit", stop the chatbot
                    if text.lower() in ["exit", "quit", "stop"]:
                        print("Exiting timelith Support Bot...")
                        self.is_recording = False
                        break
                    
                    # Get response from Gemini
                    response = conversation.send_message(text)
                    response_text = response.text
                    print(f"timelith Support Bot: {response_text}")
                    
                    # Convert text to speech
                    self.text_to_speech(response_text)
                    
                except sr.UnknownValueError:
                    print("Sorry, I couldn't understand what you said.")
                    self.text_to_speech("Sorry, I couldn't understand what you said. Could you please repeat your question about timelith?")
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
        print("Starting timelith Support Bot...")
        print("Ask questions about timelith timetable generation suite.")
        print("Say 'exit', 'quit', or 'stop' to end the conversation.")
        
        # Provide an initial message
        initial_message = "Welcome to timelith Support. I'm your specialized assistant for the timelith online timetable generation suite. How can I help you today with timelith?"
        print(f"timelith Support Bot: {initial_message}")
        self.text_to_speech(initial_message)
        
        try:
            # Start recording in a separate thread
            self.start_recording()
            
            # Process audio in the main thread
            self.process_audio()
        except KeyboardInterrupt:
            print("\nStopping the timelith Support Bot...")
        finally:
            self.stop_recording()
            print("timelith Support Bot stopped.")

if __name__ == "__main__":
    support_bot = timelithSupportBot()
    support_bot.run()