import streamlit as st
import os
from langchain_ollama import ChatOllama
from langchain_pinecone import Pinecone as LangchainPinecone
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pinecone import Pinecone
from dotenv import load_dotenv
import time

load_dotenv()

# ---- Page Configuration ---- #
st.set_page_config(
    page_title="PDPC Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Custom CSS ---- #
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# ---- Header ---- #
st.markdown('<p class="main-header">🤖 PDPC Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Your AI-powered guide to PDPC regulations and guidelines</p>', unsafe_allow_html=True)

# ---- Session State Setup (Initialize Early) ---- #
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    # Add welcome message
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": "👋 Hello! I'm your PDPC Assistant. I can help you understand PDPC regulations and guidelines. What would you like to know?"
    })

if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = False

if "show_sources" not in st.session_state:
    st.session_state.show_sources = {}

# ---- Sidebar ---- #
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model selection with descriptions
    st.subheader("AI Model")
    model_info = {
        "llama3:8b": "🚀 Fast & Accurate (Recommended)",
        "deepseek-r1:1.5b": "⚡ Lightweight & Quick"
    }
    MODEL = st.selectbox(
        "Choose a Model",
        options=list(model_info.keys()),
        format_func=lambda x: f"{x} - {model_info[x]}",
        index=0
    )
    
    st.divider()
    
    # Chat settings
    st.subheader("Chat Settings")
    MAX_HISTORY = st.slider(
        "Conversation Memory",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of previous messages to remember"
    )
    
    SHOW_SOURCES = st.checkbox(
        "Show source documents",
        value=True,
        help="Display the documents used to generate answers"
    )
    
    st.divider()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
    
    st.divider()
    
    # Chat History Section
    st.subheader("💬 Chat History")
    
    if len(st.session_state.chat_history) > 1:  # More than just welcome message
        # Container for chat history with scrolling
        history_container = st.container(height=400)
        
        with history_container:
            for i, msg in enumerate(st.session_state.chat_history):
                if msg["role"] == "user":
                    st.markdown(f"**🧑 You:** {msg['content'][:100]}{'...' if len(msg['content']) > 100 else ''}")
                else:
                    st.markdown(f"**🤖 Bot:** {msg['content'][:100]}{'...' if len(msg['content']) > 100 else ''}")
                
                if i < len(st.session_state.chat_history) - 1:
                    st.markdown("---")
    else:
        st.info("No chat history yet. Start a conversation!")
    
    st.divider()
    
    # Info section
    with st.expander("ℹ️ About"):
        st.markdown("""
        This chatbot uses:
        - **RAG** (Retrieval-Augmented Generation)
        - **Pinecone** for vector storage
        - **Ollama** for local AI models
        
        Ask questions about PDPC regulations, guidelines, and best practices!
        """)

# ---- Initialize Components ---- #
@st.cache_resource
def initialize_components():
    """Initialize LangChain components with caching"""
    try:
        # Pinecone Setup
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        if not PINECONE_API_KEY:
            return None, "❌ Pinecone API key not found. Please check your .env file."
        
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index("chatbot-index")
        
        # Embeddings
        embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        
        # Vectorstore
        try:
            vectorstore = LangchainPinecone(index=index, embedding=embeddings, text_key="text")
        except:
            vectorstore = LangchainPinecone.from_existing_index(
                index_name="chatbot-index",
                embedding=embeddings
            )
        
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        
        return retriever, None
    except Exception as e:
        return None, f"❌ Initialization error: {str(e)}"

# Initialize components
with st.spinner("🔄 Initializing AI components..."):
    retriever, error = initialize_components()

if error:
    st.error(error)
    st.stop()

# ---- LangChain Setup ---- #
llm = ChatOllama(model=MODEL, temperature=0.7)

# Enhanced RAG prompt
template = """You are a helpful PDPC (Personal Data Protection Commission) assistant. Answer the question based on the provided context.

Context:
{context}

Question: {question}

Instructions:
- Provide clear, accurate answers based on the context
- If the context doesn't contain enough information, say so honestly
- Use a friendly, professional tone
- Format your answer with bullet points or numbered lists when appropriate

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    """Format documents for context"""
    return "\n\n".join(doc.page_content for doc in docs)

# Create RAG chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ---- Display Chat History ---- #
for idx, msg in enumerate(st.session_state.chat_history):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Show sources for assistant messages if available
        if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
            st.markdown("---")
            st.markdown("### 📚 Sources Used")
            
            for i, source in enumerate(msg["sources"], 1):
                with st.expander(f"📄 Source {i}", expanded=False):
                    st.markdown("**Content:**")
                    st.text_area(
                        "Document Content",
                        source["content"] + "...",
                        height=100,
                        key=f"history_source_{idx}_{i}",
                        label_visibility="collapsed"
                    )
                    
                    if source.get("metadata"):
                        st.markdown("**Metadata:**")
                        for key, value in source["metadata"].items():
                            st.markdown(f"- **{key}:** {value}")

# ---- Trim Chat Memory ---- #
def trim_memory():
    """Keep only recent messages"""
    # Keep welcome message + recent history
    if len(st.session_state.chat_history) > (MAX_HISTORY * 2 + 1):
        welcome_msg = st.session_state.chat_history[0]
        st.session_state.chat_history = [welcome_msg] + st.session_state.chat_history[-(MAX_HISTORY * 2):]

# ---- Suggested Questions ---- #
if len(st.session_state.chat_history) <= 1:  # Only show on first load
    st.markdown("### 💡 Suggested Questions")
    col1, col2, col3 = st.columns(3)
    
    suggestions = [
        "What is PDPC?",
        "What are personal data protection obligations?",
        "How to handle data breaches?"
    ]
    
    for col, suggestion in zip([col1, col2, col3], suggestions):
        with col:
            if st.button(suggestion, use_container_width=True):
                st.session_state.pending_question = suggestion
                st.rerun()

# Handle pending question from suggestion buttons
if "pending_question" in st.session_state:
    prompt_input = st.session_state.pending_question
    del st.session_state.pending_question
else:
    prompt_input = st.chat_input("💬 Ask me anything about PDPC...")

# ---- Handle User Input ---- #
if prompt_input:
    # Add user message
    st.session_state.chat_history.append({"role": "user", "content": prompt_input})
    
    with st.chat_message("user"):
        st.markdown(prompt_input)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            try:
                # Get source documents first
                source_docs = retriever.invoke(prompt_input)
                
                # Get response with streaming effect
                full_response = rag_chain.invoke(prompt_input)
                
                # Display response with typing effect
                response_placeholder = st.empty()
                displayed_response = ""
                for chunk in full_response.split():
                    displayed_response += chunk + " "
                    response_placeholder.markdown(displayed_response + "▌")
                    time.sleep(0.02)
                response_placeholder.markdown(full_response)
                
                # Always display source documents inline
                if source_docs:
                    st.markdown("---")
                    st.markdown("### 📚 Sources Used")
                    
                    for i, doc in enumerate(source_docs, 1):
                        with st.expander(f"📄 Source {i}", expanded=False):
                            # Show preview
                            st.markdown("**Content:**")
                            content_preview = doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else "")
                            st.text_area(
                                "Document Content",
                                content_preview,
                                height=150,
                                key=f"source_{i}_{len(st.session_state.chat_history)}",
                                label_visibility="collapsed"
                            )
                            
                            # Show metadata if available
                            if hasattr(doc, 'metadata') and doc.metadata:
                                st.markdown("**Metadata:**")
                                # Format metadata nicely
                                for key, value in doc.metadata.items():
                                    st.markdown(f"- **{key}:** {value}")
                
                # Save to history with sources
                message_entry = {
                    "role": "assistant",
                    "content": full_response,
                    "sources": [{"content": doc.page_content[:200], "metadata": doc.metadata if hasattr(doc, 'metadata') else {}} 
                               for doc in source_docs] if source_docs else []
                }
                st.session_state.chat_history.append(message_entry)
                
            except Exception as e:
                error_msg = f"❌ Sorry, I encountered an error: {str(e)}"
                st.error(error_msg)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": error_msg
                })

    trim_memory()
    st.rerun()

# ---- Footer ---- #
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666; font-size: 0.9rem;'>"
    "💡 Tip: Use the sidebar to adjust settings and clear chat history"
    "</p>",
    unsafe_allow_html=True
)