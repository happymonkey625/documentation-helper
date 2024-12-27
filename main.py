# Import necessary libraries
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from typing import Set
import os

import streamlit as st
from streamlit_chat import message

from backend.core import run_llm

# Configure the Streamlit page settings
st.set_page_config(
    page_title="Your App Title",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Import libraries for handling images and HTTP requests
from PIL import Image
import requests
from io import BytesIO


# Function to format the source URLs into a readable string
def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source}\n"
    return sources_string


# Function to fetch and generate user profile picture using Gravatar
def get_profile_picture(email):
    # This uses Gravatar to get a profile picture based on email
    # You can replace this with a different service or use a default image
    gravatar_url = f"https://www.gravatar.com/avatar/{hash(email)}?d=identicon&s=200"
    response = requests.get(gravatar_url)
    img = Image.open(BytesIO(response.content))
    return img


# Load and apply external CSS
with open('static/styles.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Create sidebar with user profile information
with st.sidebar:
    st.title("User Profile")

    # Placeholder user data - can be replaced with actual user data
    user_name = "John Doe"
    user_email = "john.doe@example.com"

    profile_pic = get_profile_picture(user_email)
    st.image(profile_pic, width=150)
    st.write(f"**Name:** {user_name}")
    st.write(f"**Email:** {user_email}")

# Main app header
st.header("LangChain🦜🔗 - Helper Bot")

# Initialize session state variables to store chat history
if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_history"] = []

# Create a two-column layout for input field and submit button
col1, col2 = st.columns([2, 1])

with col1:
    prompt = st.text_input("Prompt", placeholder="Enter your message here...")

with col2:
    if st.button("Submit", key="submit"):
        prompt = prompt or "Hello"  # Set default message if input is empty

# Process user input and generate response
if prompt:
    with st.spinner("Generating response..."):
        # Get response from LLM
        generated_response = run_llm(
            query=prompt, chat_history=st.session_state["chat_history"]
        )

        # Extract and format sources from the response
        sources = set(doc.metadata["source"] for doc in generated_response["context"])
        formatted_response = (
            f"{generated_response['answer']} \n\n {create_sources_string(sources)}"
        )

        # Update session state with new messages
        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)
        st.session_state["chat_history"].append(("human", prompt))
        st.session_state["chat_history"].append(("ai", generated_response["answer"]))

# Display the chat history with alternating user and bot messages
if st.session_state["chat_answers_history"]:
    for generated_response, user_query in zip(
        st.session_state["chat_answers_history"],
        st.session_state["user_prompt_history"],
    ):
        message(user_query, is_user=True, key=f"user_{user_query}")
        message(generated_response, key=f"bot_{generated_response}")

# Add footer with attribution
st.markdown("---")
st.markdown("Powered by LangChain and Streamlit")
