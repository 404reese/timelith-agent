# app.py
import streamlit as st
from groq import Groq
import os

# App configuration
st.set_page_config(
    page_title="Timetable Analyzer",
    page_icon="📅",
    layout="wide"
)

# Initialize Groq client
def get_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY not found in secrets or environment variables")
        return None
    return Groq(api_key=api_key)

# Analysis function
def analyze_timetable(client, text):
    system_prompt = """You are an expert academic scheduler. Analyze this timetable score explanation and provide:
    1. Constraints breakdown (hard/medium/soft)
    2. Top 3 issues with counts
    3. Specific improvement recommendations
    4. Overall quality assessment
    
    Use markdown formatting with headings, bullet points, and emojis."""
    
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Analyze this timetable:\n{text}"}
        ],
        model="mixtral-8x7b-32768",
        temperature=0.3,
        max_tokens=1024
    )
    return response.choices[0].message.content

# Main UI
def main():
    st.title("📅 Timetable Analysis Tool")
    st.markdown("Paste your timetable score explanation below for analysis")
    
    # Input section
    with st.form("analysis_form"):
        input_text = st.text_area(
            "Paste your score explanation here:",
            height=300,
            placeholder="Paste your timetable score explanation here..."
        )
        
        analyze_btn = st.form_submit_button("Analyze Timetable", type="primary")
    
    # Sample data button
    if st.button("Load Example Data"):
        st.session_state.example_data = """
        [PASTE YOUR SAMPLE DATA HERE]
        """  # Put the sample data here
        st.experimental_rerun()
    
    # Display analysis results
    if 'analysis_result' in st.session_state:
        st.divider()
        st.markdown("### Analysis Results")
        st.markdown(st.session_state.analysis_result)
    
    # Handle form submission
    if analyze_btn and input_text:
        if len(input_text) < 50:
            st.warning("Please provide a longer score explanation for meaningful analysis")
            return
            
        client = get_groq_client()
        if not client:
            return
            
        with st.spinner("Analyzing timetable..."):
            try:
                result = analyze_timetable(client, input_text)
                st.session_state.analysis_result = result
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    main()