# app.py
from flask import Flask, render_template, request, jsonify, make_response
import os
import google.generativeai as genai
from dotenv import load_dotenv
import markdown
import pdfkit

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure Gemini API
def configure_genai():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return False
    genai.configure(api_key=api_key)
    return True

# Initialize the  model
def get_gemini_response(text_prompt):
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    prompt = f"""
    Analyze the following schedule data and generate a schedule evaluation report in this exact format:

    SCHEDULE EVALUATION REPORT 
    Generation: 
    Date/Time: 
    CRITICAL ISSUES 
    [Identify any critical scheduling issues that require immediate attention]
    
    SCHEDULE STATUS: [Acceptable/Unacceptable]
    
    IDENTIFIED PROBLEMS
    • [List all identified scheduling problems as bullet points]
    
    VALID SCHEDULE ELEMENTS
    • [List all valid elements of the schedule as bullet points]
    
    RECOMMENDED ACTIONS
    1. Priority Resolution
    • [List high priority actions needed]
    
    2. Secondary Adjustments
    • [List secondary adjustments if needed]
    
    NEXT STEPS
    • [List recommended next steps as bullet points]

    Schedule data to analyze:
    {text_prompt}
    """
    
    response = model.generate_content(prompt)
    return response.text

@app.route('/')
def index():
    api_configured = configure_genai()
    return render_template('index.html', api_configured=api_configured)

@app.route('/analyze', methods=['POST'])
def analyze():
    if not configure_genai():
        return jsonify({'error': 'Gemini API key not configured'}), 400
    
    text = request.form.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        analysis = get_gemini_response(text)
        html_content = markdown.markdown(analysis)
        return jsonify({'analysis': analysis, 'html_content': html_content})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download-pdf', methods=['POST'])
def download_pdf():
    try:
        html_content = request.form.get('html_content', '')
        if not html_content:
            return jsonify({'error': 'No content provided'}), 400
        
        # Create a styled HTML document for PDF conversion
        styled_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Text Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #3498db; margin-top: 20px; }}
                h3 {{ color: #2980b9; }}
                p {{ line-height: 1.6; }}
                ul, ol {{ margin-left: 20px; }}
                .container {{ max-width: 800px; margin: 0 auto; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Text Analysis Report</h1>
                {html_content}
            </div>
        </body>
        </html>
        """
        
        # Convert HTML to PDF
        pdf = pdfkit.from_string(styled_html, False)
        
        # Create response with PDF
        response = make_response(pdf)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = 'attachment; filename=text_analysis_report.pdf'
        
        return response
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)