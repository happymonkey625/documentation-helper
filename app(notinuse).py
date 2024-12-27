import streamlit as st
from backend.core import run_llm2

# Set page configuration
st.set_page_config(
    page_title="LangChain Documentation Assistant",
    page_icon="🦜",
    layout="wide"
)

# Add header
st.title("🦜 LangChain Documentation Assistant")
st.markdown("""
This app uses RAG (Retrieval Augmented Generation) to answer questions about LangChain using the official documentation.
Ask any question about LangChain, and the assistant will find relevant documentation to answer your query.
""")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about LangChain"):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching documentation..."):
            response = run_llm2(prompt)
            st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response}) 