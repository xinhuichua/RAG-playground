import streamlit as st
import os
import time
import re
import uuid
from datetime import datetime, UTC
from dotenv import load_dotenv

# LangChain / RAG bits
# from langchain_ollama import ChatOllama
# from langchain_pinecone import Pinecone as LangchainPinecone
# from langchain_ollama.embeddings import OllamaEmbeddings
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# from pinecone import Pinecone

# Project helpers
from supabase_config import (
    initialize_supabase,
    save_chat_message,
    load_chat_history,
)
from auth_supabase import show_login_page, logout

load_dotenv()
st.set_page_config(page_title="PDPC Assistant", page_icon="🤖", layout="wide")

# ---------- Session defaults ----------
st.session_state.setdefault("authenticated", False)
st.session_state.setdefault("chat_loaded", False)
st.session_state.setdefault("messages", [])           # list of {role, content, thread_id, created_at}
st.session_state.setdefault("threads", {})            # {thread_id: {title, created_at}}
st.session_state.setdefault("active_thread_id", None) # current open thread
st.session_state.setdefault("last_save_status", "—")

def utc_now_iso():
    return datetime.now(UTC).isoformat()

# ---------- Auth ----------
if not st.session_state.authenticated:
    show_login_page()
    st.stop()

# ---------- Supabase ----------
if "supabase" not in st.session_state:
    st.session_state.supabase = initialize_supabase()

supabase = st.session_state.supabase
if not supabase:
    st.error("❌ Failed to connect to Supabase. Please check configuration.")
    st.stop()

# ---------- Verify Authentication & Get User ID ----------
try:
    auth_user = supabase.auth.get_user()
    if auth_user and auth_user.user:
        st.session_state.user_id = auth_user.user.id
        st.session_state.user_email = auth_user.user.email
    else:
        st.error("❌ Authentication session expired. Please log in again.")
        logout()
        st.stop()
except Exception as e:
    st.error(f"❌ Could not verify authentication: {e}")
    logout()
    st.stop()

# ---------- Helpers ----------
def summarize_title(text: str, max_chars: int = 60) -> str:
    if not text:
        return "Untitled"
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    for sep in ["?", ".", "!", "\n", "•", ";", ":"]:
        if sep in text:
            text = text.split(sep)[0]
            break
    if len(text) > max_chars:
        text = text[:max_chars - 1].rstrip() + "…"
    return text[:1].upper() + text[1:] if text else "Untitled"

def persist_message(role: str, content: str, thread_id: str = None, title: str = None):
    """Save a message to Supabase and log status in UI."""
    try:
        ok, err = save_chat_message(
            supabase, 
            st.session_state.user_id, 
            role, 
            content, 
            thread_id=thread_id, 
            title=title
        )
        if ok:
            st.session_state.last_save_status = f"✅ Saved at {datetime.now(UTC).strftime('%H:%M:%S')}"
            st.toast("Message saved", icon="✅")
        else:
            st.session_state.last_save_status = f"⚠️ Save failed: {err}"
            st.toast("Save failed", icon="⚠️")
    except Exception as e:
        st.session_state.last_save_status = f"❌ Save error: {e}"
        st.toast("Save error", icon="❌")

def new_thread() -> str:
    thread_id = str(uuid.uuid4())
    st.session_state.threads[thread_id] = {
        "title": None,
        "created_at": utc_now_iso(),
    }
    st.session_state.active_thread_id = thread_id
    # Friendly greeting (display only, not saved)
    msg = {
        "role": "assistant",
        "content": "Hi! I'm your PDPC Assistant. What would you like to know?",
        "thread_id": thread_id,
        "created_at": utc_now_iso(),
        "temp": True  # Mark as temporary/not persisted
    }
    st.session_state.messages.append(msg)
    # Don't save the greeting message
    return thread_id

# ---------- Load persisted messages once ----------
if not st.session_state.chat_loaded:
    with st.spinner("Loading your chat…"):
        records, err = load_chat_history(supabase, st.session_state.user_id, limit=1000)
        if err:
            st.warning("Couldn't load previous chats; starting fresh.")
            records = []
        threads_seen = {}
        for r in records or []:
            role = r.get("role", "assistant")
            content = r.get("content", "")
            thread_id = r.get("thread_id") or r.get("chat_id") or "default"
            created_at = r.get("created_at") or utc_now_iso()
            title = r.get("title")
            st.session_state.messages.append({
                "role": role, "content": content, "thread_id": thread_id, "created_at": created_at
            })
            if thread_id not in threads_seen:
                threads_seen[thread_id] = {"title": title, "created_at": created_at}
        for tid, meta in threads_seen.items():
            if not meta["title"]:
                first_user = next((m for m in st.session_state.messages if m["thread_id"]==tid and m["role"]=="user"), None)
                meta["title"] = summarize_title(first_user["content"]) if first_user else f"Chat {tid[:8]}"
        st.session_state.threads.update(threads_seen)
        if st.session_state.threads:
            st.session_state.active_thread_id = sorted(
                st.session_state.threads.items(),
                key=lambda kv: kv[1]["created_at"]
            )[-1][0]
        else:
            new_thread()
        st.session_state.chat_loaded = True

# ---------- RAG bootstrap (cached) ----------
@st.cache_resource
def _init_retriever():
    try:
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            return None, "Missing PINECONE_API_KEY"
        pc = Pinecone(api_key=api_key)
        index = pc.Index("chatbot-index")
        embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        try:
            vs = LangchainPinecone(index=index, embedding=embeddings, text_key="text")
        except Exception:
            vs = LangchainPinecone.from_existing_index(index_name="chatbot-index", embedding=embeddings)
        retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        return retriever, None
    except Exception as e:
        return None, str(e)

retriever, _err = _init_retriever()
if _err:
    st.info("RAG search unavailable right now; answering without retrieved docs.")

MODEL = "llama3.2:1b"  # Changed to smaller model to avoid GPU memory issues
llm = ChatOllama(model=MODEL, temperature=0.3, num_ctx=2048)

template = """You are a helpful PDPC assistant. Use the provided context if available.
Context:
{context}

Question: {question}

Guidelines:
- Be accurate and concise.
- If the context is insufficient, say so.
- Use short bullet points where helpful.

Answer:"""
prompt = ChatPromptTemplate.from_template(template)

def _fmt(docs):
    return "\n\n".join(getattr(d, "page_content", "") for d in docs) if docs else ""

rag_chain = (
    {"context": (retriever if retriever else RunnablePassthrough()) | _fmt, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ---------- Sidebar: threads (clickable) ----------
with st.sidebar:
    st.markdown(f"**Signed in as:** {st.session_state.get('user_email', 'User')}")
    st.divider()
    st.subheader("Chats")
    if st.button("➕ New chat", use_container_width=True):
        new_thread()
        st.rerun()
    ordered_threads = sorted(st.session_state.threads.items(), key=lambda kv: kv[1]["created_at"], reverse=True)
    for tid, meta in ordered_threads:
        label = meta.get("title") or f"Chat {tid[:8]}"
        if st.button(label, key=f"threadbtn-{tid}", use_container_width=True):
            st.session_state.active_thread_id = tid
            st.rerun()
    st.caption(f"Last save: {st.session_state.last_save_status}")
    st.divider()
    if st.button("Logout", use_container_width=True):
        logout()

# ---------- Main: render only active thread ----------
active_tid = st.session_state.active_thread_id or new_thread()

active_msgs = [m for m in st.session_state.messages if m["thread_id"] == active_tid]
if not active_msgs:
    msg = {
        "role": "assistant",
        "content": "Hi! I'm your PDPC Assistant. What would you like to know?",
        "thread_id": active_tid,
        "created_at": utc_now_iso(),
        "temp": True  # Mark as temporary
    }
    st.session_state.messages.append(msg)
    # Don't save the greeting message
    active_msgs = [m for m in st.session_state.messages if m["thread_id"] == active_tid]

st.title("🤖 PDPC Assistant")
st.caption("Ask anything about PDPC. Chats auto-save and are grouped by thread in the sidebar.")

for msg in active_msgs:
    with st.chat_message(msg["role"]):
        st.markdown(msg.get("content", ""))

# ---------- Input ----------
user_text = st.chat_input("Ask about PDPC…")

# ---------- Handle turn within the active thread ----------
if user_text:
    current = st.session_state.threads.get(active_tid, {"title": None, "created_at": utc_now_iso()})
    placeholder_titles = {None, "", "New chat", f"Chat {active_tid[:8]}"}
    if current.get("title") in placeholder_titles:
        new_title = summarize_title(user_text)
        current["title"] = new_title
        st.session_state.threads[active_tid] = current
        # Log a meta save (optional) so you can audit title updates in DB logs if you add support later
        st.session_state.last_save_status = f"🛈 Title set: {new_title}"

    # Show & save user message
    st.session_state.messages.append({"role": "user", "content": user_text, "thread_id": active_tid, "created_at": utc_now_iso()})
    persist_message("user", user_text, thread_id=active_tid, title=current.get("title"))

    with st.chat_message("user"):
        st.markdown(user_text)

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                response = rag_chain.invoke(user_text)
            except Exception as e:
                response = f"Sorry, I hit an error answering that: {e}"

            ph = st.empty()
            sofar = ""
            for token in response.split():
                sofar += token + " "
                ph.markdown(sofar + "▌")
                time.sleep(0.015)
            ph.markdown(response)

            st.session_state.messages.append({"role": "assistant", "content": response, "thread_id": active_tid, "created_at": utc_now_iso()})
            persist_message("assistant", response, thread_id=active_tid, title=current.get("title"))

    st.rerun()

st.markdown("---")
st.caption("💾 Chats auto-save to your account and are organized into threads.")