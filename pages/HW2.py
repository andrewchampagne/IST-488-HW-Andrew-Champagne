# pages/lab2.py
import streamlit as st
from openai import OpenAI
from groq import Groq
import requests
from bs4 import BeautifulSoup
# Show title and description.
st.title("Lab 2")
st.write(
    "Enter a URL below and ask a question about it – GPT will answer!"
)

# Sidebar options
st.sidebar.header("Summary Options")

summary_type = st.sidebar.radio(
    "Choose a summary format:",
    [
        "Summarize in 100 words",
        "Summarize in 2 connecting paragraphs",
        "Summarize in 5 bullet points"
    ]
)

output_language = st.sidebar.selectbox(
    "Output language:",
    ["English", "French", "Spanish", "German", "Chinese"]
)

llm_provider = st.sidebar.selectbox(
    "Choose LLM:",
    ["OpenAI", "Groq"]
)

use_advanced_model = st.sidebar.checkbox("Use advanced model")

if llm_provider == "OpenAI":
    model = "gpt-4o" if use_advanced_model else "gpt-4o-mini"
else:  # Groq
    model = "llama-3.3-70b-versatile" if use_advanced_model else "llama-3.1-8b-instant"

st.sidebar.write(f"Current model: {model}")

# Get API key from Streamlit secrets
# Also made a toml file
openai_api_key = st.secrets["OPENAI_API_KEY"]
groq_api_key = st.secrets["GROQ_API_KEY"]

try:
    # Create the appropriate client based on provider
    if llm_provider == "OpenAI":
        client = OpenAI(api_key=openai_api_key)
    else:
        client = Groq(api_key=groq_api_key)
    
    # Let the user enter a URL
    url = st.text_input("Enter a URL:")
    
    document = None
    if url:
        try:
            response = requests.get(url)
            response.raise_for_status() # Raise an exception for HTTP errors
            soup = BeautifulSoup(response.content, 'html.parser')
            document = soup.get_text()
            st.success("URL loaded successfully!")
        except requests.RequestException as e:
            st.error(f"Error reading {url}: {e}")


    # Generate summary button because there is no longer a need for a text area 
    if st.button("Generate Summary", disabled=not document):

            

            
            # Build the prompt based on summary type
            if summary_type == "Summarize in 100 words":
                summary_instruction = f"Please summarize this document in approximately 100 words. Output the summary in {output_language}."
            elif summary_type == "Summarize in 2 connecting paragraphs":
                summary_instruction = f"Please summarize this document in 2 connecting paragraphs. Output the summary in {output_language}."
            else:
                summary_instruction = f"Please summarize this document in 5 bullet points. Output the summary in {output_language}."
            
            messages = [
                {
                    "role": "user",
                    "content": f"Here's a document: {document} \n\n---\n\n{summary_instruction}",
                }
            ]

            # Generate an answer using the selected API.
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
            )
            st.write_stream(stream)
        
except Exception as e:
    st.error(f"Error: {e}")