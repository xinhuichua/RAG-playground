import streamlit as st
from supabase import create_client, Client
from datetime import datetime
import json
import os
from dotenv import load_dotenv

load_dotenv()


def initialize_supabase() -> Client:
    """Initialize Supabase client"""
    try:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        if not url or not key:
            st.error("Missing SUPABASE_URL or SUPABASE_KEY in environment")
            return None
        supabase: Client = create_client(url, key)
        return supabase
    except Exception as e:
        st.error(f"Failed to initialize Supabase: {e}")
        return None


def sign_up_user(supabase: Client, email: str, password: str, display_name: str = None):
    """Sign up a new user"""
    try:
        response = supabase.auth.sign_up({
            "email": email,
            "password": password,
            "options": {
                "data": {
                    "display_name": display_name or email.split('@')[0]
                }
            }
        })
        
        if response.user:
            # User profile is automatically created by database trigger
            return response.user, None
        else:
            return None, "Failed to create user"
    except Exception as e:
        return None, str(e)


def sign_in_user(supabase: Client, email: str, password: str):
    """Sign in an existing user"""
    try:
        response = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
        
        if response.user:
            # Update last login
            supabase.table('users').update({
                'last_login': datetime.now().isoformat()
            }).eq('id', response.user.id).execute()
            
            return response.user, response.session, None
        else:
            return None, None, "Invalid credentials"
    except Exception as e:
        return None, None, str(e)


def sign_out_user(supabase: Client):
    """Sign out current user"""
    try:
        supabase.auth.sign_out()
        return True, None
    except Exception as e:
        return False, str(e)


def reset_password(supabase: Client, email: str):
    """Send password reset email"""
    try:
        supabase.auth.reset_password_email(email)
        return True, None
    except Exception as e:
        return False, str(e)


def save_chat_message(
    supabase: Client,
    user_id: str,
    role: str,
    content: str,
    sources: list = None,
    thread_id: str = None,
    title: str = None
):
    """Save a single chat message with optional thread and title."""
    try:
        data = {
            'user_id': user_id,
            'role': role,
            'content': content,
            'thread_id': thread_id,
            'title': title,
            'created_at': datetime.now().isoformat()
        }
        
        # Only add sources if provided
        if sources:
            data['sources'] = json.dumps(sources)
        
        response = supabase.table('chat_messages').insert(data).execute()
        return True, None
    except Exception as e:
        return False, str(e)


def load_chat_history(supabase: Client, user_id: str, limit: int = 1000):
    """Load user's chat history"""
    try:
        response = supabase.table('chat_messages')\
            .select('*')\
            .eq('user_id', user_id)\
            .order('created_at', desc=False)\
            .limit(limit)\
            .execute()
        
        if response.data:
            # Parse sources JSON if present
            for msg in response.data:
                if msg.get('sources') and isinstance(msg['sources'], str):
                    try:
                        msg['sources'] = json.loads(msg['sources'])
                    except:
                        msg['sources'] = []
            return response.data, None
        else:
            return [], None
    except Exception as e:
        return [], str(e)


def save_chat_session(supabase: Client, user_id: str, session_name: str, messages: list, thread_id: str = None):
    """Save entire chat session"""
    try:
        data = {
            'user_id': user_id,
            'session_name': session_name,
            'thread_id': thread_id,
            'messages': json.dumps(messages),
            'message_count': len(messages),
            'created_at': datetime.now().isoformat()
        }
        
        response = supabase.table('chat_sessions').insert(data).execute()
        return True, None
    except Exception as e:
        return False, str(e)


def load_chat_sessions(supabase: Client, user_id: str):
    """Load all chat sessions for a user"""
    try:
        response = supabase.table('chat_sessions')\
            .select('*')\
            .eq('user_id', user_id)\
            .order('created_at', desc=True)\
            .execute()
        
        if response.data:
            # Parse messages JSON
            for session in response.data:
                if session.get('messages') and isinstance(session['messages'], str):
                    try:
                        session['messages'] = json.loads(session['messages'])
                    except:
                        session['messages'] = []
            return response.data, None
        else:
            return [], None
    except Exception as e:
        return [], str(e)


def delete_chat_history(supabase: Client, user_id: str):
    """Delete all chat messages for a user"""
    try:
        supabase.table('chat_messages').delete().eq('user_id', user_id).execute()
        return True, None
    except Exception as e:
        return False, str(e)


def delete_chat_session(supabase: Client, session_id: str):
    """Delete a specific chat session"""
    try:
        supabase.table('chat_sessions').delete().eq('id', session_id).execute()
        return True, None
    except Exception as e:
        return False, str(e)


def get_user_statistics(supabase: Client, user_id: str):
    """Get user's chat statistics"""
    try:
        # Count total messages
        messages_response = supabase.table('chat_messages')\
            .select('id', count='exact')\
            .eq('user_id', user_id)\
            .execute()
        
        # Count total sessions
        sessions_response = supabase.table('chat_sessions')\
            .select('id', count='exact')\
            .eq('user_id', user_id)\
            .execute()
        
        return {
            'total_messages': messages_response.count if messages_response.count else 0,
            'total_sessions': sessions_response.count if sessions_response.count else 0
        }, None
    except Exception as e:
        return None, str(e)


def search_chat_history(supabase: Client, user_id: str, search_term: str):
    """Search through user's chat history"""
    try:
        response = supabase.table('chat_messages')\
            .select('*')\
            .eq('user_id', user_id)\
            .ilike('content', f'%{search_term}%')\
            .order('created_at', desc=False)\
            .execute()
        
        return response.data if response.data else [], None
    except Exception as e:
        return [], str(e)


def get_user_profile(supabase: Client, user_id: str):
    """Get user profile information"""
    try:
        response = supabase.table('users')\
            .select('*')\
            .eq('id', user_id)\
            .single()\
            .execute()
        
        return response.data if response.data else None, None
    except Exception as e:
        return None, str(e)


def update_user_profile(supabase: Client, user_id: str, profile_data: dict):
    """Update user profile"""
    try:
        response = supabase.table('users')\
            .update(profile_data)\
            .eq('id', user_id)\
            .execute()
        
        return True, None
    except Exception as e:
        return False, str(e)