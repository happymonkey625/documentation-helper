# Import necessary dependencies
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from typing import Any, Dict, List
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from consts import INDEX_NAME

# Initialize Pinecone with API key
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    """
    First implementation of the RAG (Retrieval Augmented Generation) chain using a more traditional approach.
    This function processes queries while taking into account chat history for context-aware responses.
    
    Args:
        query (str): The user's input question
        chat_history (List[Dict[str, Any]]): Previous conversation history
    """
    # Initialize OpenAI embeddings with the latest embedding model
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    # Set up Pinecone vector store for document storage and retrieval
    index = pc.Index(name=INDEX_NAME)
    docsearch = PineconeVectorStore(index=index, embedding=embeddings)
    # Initialize ChatOpenAI with no temperature for deterministic outputs
    chat = ChatOpenAI(verbose=True, temperature=0)

    # Get the prompt template for query rephrasing
    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    # Get the prompt template for retrieval QA
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    
    # Create a chain to combine retrieved documents with the prompt
    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)

    # Create a retriever that's aware of conversation history
    history_aware_retriever = create_history_aware_retriever(
        llm=chat, retriever=docsearch.as_retriever(), prompt=rephrase_prompt
    )
    
    # Create the final QA chain combining the retriever and document chain
    qa = create_retrieval_chain(
        retriever=history_aware_retriever, combine_docs_chain=stuff_documents_chain
    )

    # Process the query and return results
    result = qa.invoke(input={"input": query, "chat_history": chat_history})
    return result


def format_docs(docs):
    """
    Helper function to format retrieved documents into a single string.
    
    Args:
        docs: List of documents with page_content attribute
    Returns:
        str: Concatenated string of document contents
    """
    return "\n\n".join(doc.page_content for doc in docs)


def run_llm2(query: str, chat_history: List[Dict[str, Any]] = []):
    """
    Second implementation of the RAG chain using a more modern LCEL (LangChain Expression Language) approach.
    This implementation provides more flexibility and composability.
    
    Args:
        query (str): The user's input question
        chat_history (List[Dict[str, Any]]): Previous conversation history
    """
    # Initialize embeddings and vector store
    embeddings = OpenAIEmbeddings()
    index = pc.Index(name=INDEX_NAME)
    docsearch = PineconeVectorStore(index=index, embedding=embeddings)
    
    # Initialize ChatGPT-4 for more advanced reasoning
    chat = ChatOpenAI(model_name="gpt-4", verbose=True, temperature=0)

    # Get the retrieval QA prompt template
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    # Create a simple retrieval chain
    retriever = docsearch.as_retriever()
    
    # Create the RAG chain
    chain = (
        {"context": retriever, "input": RunnablePassthrough()}
        | retrieval_qa_chat_prompt
        | chat
        | StrOutputParser()
    )

    # Process the query and return results
    return chain.invoke(query)
