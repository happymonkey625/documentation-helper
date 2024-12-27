import os
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from pinecone import Pinecone
from consts import INDEX_NAME

# Initialize Pinecone client
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# Create index if it doesn't exist
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,  # OpenAI embedding dimension
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )

def ingest_docs() -> None:
    # Update the path to match your actual documentation location
    path = "langchain-docs/api.python.langchain.com/en/latest"
    print(f"Loading documents from: {path}")
    
    try:
        loader = ReadTheDocsLoader(path=path)
        raw_documents = loader.load()
        if not raw_documents:
            raise ValueError(f"No documents found in {path}. Please check if the path exists and contains .html files")
            
        print(f"Loaded {len(raw_documents)} documents")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100, 
            separators=["\n\n", "\n", " ", ""]
        )
        documents = text_splitter.split_documents(documents=raw_documents)
        print(f"Splitted into {len(documents)} chunks")

        for doc in documents:
            old_path = doc.metadata["source"]
            new_url = old_path.replace("langchain-docs", "https:/")
            doc.metadata.update({"source": new_url})

        print(f"Going to insert {len(documents)} to Pinecone")
        embeddings = OpenAIEmbeddings()
        LangchainPinecone.from_documents(documents, embeddings, index_name=INDEX_NAME)
        print("****** Added to Pinecone vectorstore vectors")
    except Exception as e:
        print(f"Error loading documents: {e}")
        raise

if __name__ == "__main__":
    ingest_docs()
