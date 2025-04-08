from flask import Flask, request, jsonify
import os
from groq import Groq

app = Flask(__name__)

# Load Groq API key from environment variable or request argument
GROQ_API_KEY = request.args.get("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")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set or api_key argument not provided")

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

@app.route('/')
def home():
    return "Groq JSON Report Generator API is running!"

@app.route('/generate_report', methods=['POST'])
def generate_report():
    if not request.is_json:
        return jsonify({"error": "Request must be in JSON format"}), 400
    data = request.get_json()

    try:
        user_name = data.get("user_name", "User")
        topic = data.get("topic", "no topic provided")
        details = data.get("details", {})

        # Create a prompt for the LLM
        prompt = f"""
        Please analyze the following score explanation and provide a clear, concise summary for the user:

    Score Explanation:
 {score_explanation}

    The summary should include:
    - A brief overview of the score
    - A breakdown of the constraints and their impact on the score
        - Insights into the indictments and their effects
    - Recommendations or suggestions for improvement

Provide the summary in a format that is easy for a non-technical user to understand.
        """

        # Call Groq LLM (using Mixtral model)
        response = client.chat.completions.create(
            model="mixtral-8x7b-32768",  # or llama3-8b-8192 / gemma-7b-it
            messages=[
                {"role": "system", "content": "You are a helpful report generator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        llm_reply = response.choices[0].message.content

        return jsonify({
            "status": "success",
            "generated_report": llm_reply
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

