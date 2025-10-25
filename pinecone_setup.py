
#load the pdfs into Pinecone DB
#Chunking and emebdding with Ollama Embeddings


from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# Initialize Pinecone (new API)
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "chatbot-index"

# Create an index if it doesn't exist
existing_indexes = [index.name for index in pc.list_indexes()]
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=1024,  # Use 1024 for mxbai-embed-large
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)
print("Pinecone index initialized!")

# Load PDF documents from data folder
from langchain_community.document_loaders import PyPDFLoader

def load_pdf_documents(folder_path):
    """Load all PDF files from a folder"""
    documents = []
    pdf_files = Path(folder_path).glob("*.pdf")
    
    for pdf_file in pdf_files:
        print(f"Loading {pdf_file.name}...")
        loader = PyPDFLoader(str(pdf_file))
        documents.extend(loader.load())
    
    return documents

# Load all PDFs from data folder
data_folder = "data/"
pdf_docs = load_pdf_documents(data_folder)
print(f"Loaded {len(pdf_docs)} pages from PDFs in {data_folder}")

# Split documents into chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)

splited_documents = text_splitter.split_documents(pdf_docs)
print(f"Split into {len(splited_documents)} chunks")

# Create embeddings and vectorstore
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

embeddings = OllamaEmbeddings(model='mxbai-embed-large')

vectorstore = PineconeVectorStore(embedding=embeddings, index=index)

# Add documents to vectorstore
vectorstore.add_documents(documents=splited_documents)
print(f"Added {len(splited_documents)} document chunks to Pinecone!")