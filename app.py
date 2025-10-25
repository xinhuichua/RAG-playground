import streamlit as st
import os
from langchain_ollama import ChatOllama
from langchain_pinecone import Pinecone as LangchainPinecone  # Updated import
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

# ---- Pinecone Setup ---- #
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("chatbot-index")

# ---- Streamlit UI ---- #
st.set_page_config(layout="wide")
st.title("PDPC Chatbot")


st.sidebar.header("Settings")
MODEL = st.sidebar.selectbox("Choose a Model", ["llama3:8b", "deepseek-r1:1.5b"], index=0)
MAX_HISTORY = st.sidebar.number_input("Max History", 1, 10, 2)

# ---- Session State Setup ---- #
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---- LangChain Components ---- #
llm = ChatOllama(model=MODEL)
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Try different initialization approaches
try:
    # Method 1: Using LangchainPinecone
    vectorstore = LangchainPinecone(index=index, embedding=embeddings, text_key="text")
except:
    try:
        # Method 2: Using from_existing_index
        vectorstore = LangchainPinecone.from_existing_index(
            index_name="chatbot-index",
            embedding=embeddings
        )
    except Exception as e:
        st.error(f"Failed to initialize Pinecone vectorstore: {e}")
        st.stop()

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Create RAG prompt
template = """Answer the question based only on the following context:

{context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# Create RAG chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ---- Display Chat History ---- #
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---- Trim Chat Memory ---- #
def trim_memory():
    while len(st.session_state.chat_history) > MAX_HISTORY * 2:
        st.session_state.chat_history.pop(0)

# ---- Handle User Input ---- #
if prompt_input := st.chat_input("Say something"):
    st.session_state.chat_history.append({"role": "user", "content": prompt_input})
    
    with st.chat_message("user"):
        st.markdown(prompt_input)

    trim_memory()

    with st.chat_message("assistant"):
        response_container = st.empty()
        
        try:
            # Get response from RAG chain
            full_response = rag_chain.invoke(prompt_input)
            
            # Get source documents for display
            source_docs = retriever.invoke(prompt_input)
            
            response_container.markdown(full_response)
            
            # Show source documents
            if source_docs:
                with st.expander("📚 View Source Documents"):
                    for i, doc in enumerate(source_docs, 1):
                        st.markdown(f"**Source {i}:**")
                        st.text(doc.page_content[:300] + "...")
                        if hasattr(doc, 'metadata') and doc.metadata:
                            st.json(doc.metadata)
        except Exception as e:
            full_response = f"Error: {str(e)}"
            response_container.markdown(full_response)
            st.error(f"Detailed error: {e}")

        st.session_state.chat_history.append({"role": "assistant", "content": full_response})
        trim_memory()