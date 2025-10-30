import streamlit as st
from supabase import create_client, Client
import os
from dotenv import load_dotenv

load_dotenv()

def initialize_auth_supabase() -> Client:
    """Initialize Supabase client for authentication"""
    try:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        
        if not url or not key:
            st.error("❌ Supabase credentials not found in .env file")
            return None
        
        supabase: Client = create_client(url, key)
        return supabase
    except Exception as e:
        st.error(f"❌ Failed to initialize Supabase: {str(e)}")
        return None

import streamlit as st
from supabase_config import initialize_supabase, sign_in_user, sign_up_user

def show_login_page():
    """Display login/signup page"""
    st.title("🔐 PDPC Assistant Login")
    
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    with tab1:
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            
            if submit:
                if not email or not password:
                    st.error("Please enter both email and password")
                else:
                    supabase = initialize_supabase()
                    if supabase:
                        user, session, error = sign_in_user(supabase, email, password)
                        if user and session:
                            st.session_state.authenticated = True
                            st.session_state.user_id = user.id  # ✅ Critical: Set user_id
                            st.session_state.user_email = user.email
                            st.session_state.supabase = supabase
                            st.success("✅ Login successful!")
                            st.rerun()
                        else:
                            st.error(f"❌ Login failed: {error}")
    
    with tab2:
        with st.form("signup_form"):
            email = st.text_input("Email", key="signup_email")
            password = st.text_input("Password", type="password", key="signup_password")
            confirm_password = st.text_input("Confirm Password", type="password")
            display_name = st.text_input("Display Name (optional)")
            submit = st.form_submit_button("Sign Up")
            
            if submit:
                if not email or not password:
                    st.error("Please enter both email and password")
                elif password != confirm_password:
                    st.error("Passwords do not match")
                elif len(password) < 6:
                    st.error("Password must be at least 6 characters")
                else:
                    supabase = initialize_supabase()
                    if supabase:
                        user, error = sign_up_user(supabase, email, password, display_name)
                        if user:
                            st.success("✅ Account created! Please check your email to verify your account, then login.")
                        else:
                            st.error(f"❌ Sign up failed: {error}")

def logout():
    """Clear session and logout"""
    keys_to_keep = []  # Add any keys you want to preserve
    keys_to_clear = [k for k in st.session_state.keys() if k not in keys_to_keep]
    for key in keys_to_clear:
        del st.session_state[key]
    st.session_state.authenticated = False
    st.rerun()


def check_authentication():
    """Check if user is authenticated"""
    return st.session_state.get('authenticated', False)


def get_current_user():
    """Get current user information"""
    if check_authentication():
        return {
            'user_id': st.session_state.get('user_id'),
            'email': st.session_state.get('user_email'),
            'access_token': st.session_state.get('access_token')
        }
    return None