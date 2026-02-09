# pages/lab3.py
import streamlit as st
from openai import OpenAI
import tiktoken
import requests
from bs4 import BeautifulSoup

st.title("Lab 3: Streaming Chatbot with URL Context and Conversation Buffer")

st.sidebar.header("Chatbot Settings")

use_advanced_model = st.sidebar.checkbox("Use advanced model")
model = "gpt-4o" if use_advanced_model else "gpt-4o-mini"
st.sidebar.write(f"Current model: {model}")

openai_api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=openai_api_key)

def read_url_content(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        for script in soup(["script", "style"]):
            script.decompose()
        
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text
    except Exception as e:
        return f"Error reading URL: {str(e)}"

st.sidebar.header("URL Context")
url1 = st.sidebar.text_input("URL 1 (optional):", "")
url2 = st.sidebar.text_input("URL 2 (optional):", "")

if st.sidebar.button("Load URLs"):
    url_contents = []
    
    if url1:
        content1 = read_url_content(url1)
        if not content1.startswith("Error"):
            url_contents.append(f"Content from {url1}:\n\n{content1}")
            st.sidebar.success("URL 1 loaded")
        else:
            st.sidebar.error(f"URL 1 failed: {content1}")
    
    if url2:
        content2 = read_url_content(url2)
        if not content2.startswith("Error"):
            url_contents.append(f"Content from {url2}:\n\n{content2}")
            st.sidebar.success("URL 2 loaded")
        else:
            st.sidebar.error(f"URL 2 failed: {content2}")
    
    if url_contents:
        st.session_state.url_context = "\n\n---\n\n".join(url_contents)
        st.session_state.urls_loaded = True
    else:
        st.session_state.url_context = ""
        st.session_state.urls_loaded = False

if "messages" not in st.session_state:
    st.session_state.messages = []

if "url_context" not in st.session_state:
    st.session_state.url_context = ""

if "urls_loaded" not in st.session_state:
    st.session_state.urls_loaded = False

def count_tokens(messages, model_name="gpt-4o"):
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    num_tokens = 0
    for message in messages:
        num_tokens += 4
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
    num_tokens += 2
    return num_tokens

def apply_message_buffer(messages):
    if len(messages) <= 6:
        return messages
    return messages[-6:]

def create_system_prompt():
    base_prompt = """You are a helpful assistant that explains things in a way that a 10-year-old can understand. 
Use simple words, short sentences, and examples from everyday life.

After answering each question, always ask "Do you want more info?"

If the user says "Yes" (or anything similar like "yeah", "sure", "yep", "tell me more"):
- Provide more detailed information about the topic
- Ask again "Do you want more info?"

If the user says "No" (or anything similar like "nope", "no thanks", "I'm good"):
- Say something friendly like "Okay! What else can I help you with?"
- Wait for their next question"""
    
    if st.session_state.url_context:
        return f"""{base_prompt}

You have access to the following reference documents. Use this information to answer questions:

{st.session_state.url_context}"""
    else:
        return base_prompt

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.session_state.messages or st.session_state.url_context:
    st.sidebar.header("Token Statistics")
    
    system_prompt = create_system_prompt()
    system_tokens = count_tokens([{"role": "system", "content": system_prompt}], model)
    st.sidebar.write(f"System prompt tokens: {system_tokens}")
    
    if st.session_state.messages:
        buffered = apply_message_buffer(st.session_state.messages)
        buffered_tokens = count_tokens(buffered, model)
        
        total_sent = system_tokens + buffered_tokens
        st.sidebar.write(f"Total tokens sent to LLM: {total_sent}")

if prompt := st.chat_input("What would you like to know?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    buffered_messages = apply_message_buffer(st.session_state.messages)
    system_prompt = create_system_prompt()
    api_messages = [{"role": "system", "content": system_prompt}] + buffered_messages
    
    with st.chat_message("assistant"):
        response_container = st.empty()
        full_response = ""
        
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=api_messages,
                stream=True,
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    response_container.markdown(full_response)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Error generating response: {e}")

if st.sidebar.button("Clear Conversation"):
    st.session_state.messages = []
    st.rerun()