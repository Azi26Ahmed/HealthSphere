import streamlit as st
import json
import os
from streamlit_lottie import st_lottie
from components.dashboard import show_dashboard
from components.database import register_user, verify_user, init_db
from components.utils import load_lottie_file

# Initialize the database
init_db()

# Get absolute path to the Animation directory
animation_dir = os.path.join(os.path.dirname(__file__), "Animation")
icon_path = os.path.join(animation_dir, "icon.gif")

st.set_page_config(page_title="HealthSphere", page_icon=icon_path, layout="wide")

# Load your Lottie animation file
animation_path = os.path.join(animation_dir, "animation.json")
lottie_animation = load_lottie_file(animation_path)

# Function to show login/register page with layout
def show_welcome_page():
    col1, col2 = st.columns([1, 1])
    with col1:
        st.title("HealthSphere")
        st.write("Your Comprehensive Health Prediction Platform")
        st_lottie(lottie_animation, height=200, width=300, speed=10)
    with col2:
        choice = st.radio("", ["Login", "Register"], horizontal=True)
        if choice == "Login":
            show_login_page()
        else:
            show_register_page()

# Login Page
def show_login_page():
    st.subheader("Login to your Account")
    username = st.text_input("Username", placeholder="Enter your username")
    password = st.text_input("Password", placeholder="Enter your password", type="password")

    if st.button("Login"):
        success, user = verify_user(username, password)
        if success:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("Login successful! ")
            st.rerun()
        else:
            st.error("Invalid username or password.")

# Registration Page
def show_register_page():
    st.subheader("Create a New Account")
    
    username = st.text_input("Choose a Username", placeholder="Enter a unique username")
    password = st.text_input("Choose a Password", placeholder="Enter a strong password", type="password")
    confirm_password = st.text_input("Confirm Password", placeholder="Confirm your password", type="password")

    if st.button("Register"):
        if not username  or not password:
            st.error("Please fill in all fields.")
            return
            
        if password != confirm_password:
            st.error("Passwords do not match.")
            return
            
        success, message = register_user(username, password)
        if success:
            st.success(f"{message}  Please login.")
        else:
            st.error(message)

def main():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        show_welcome_page()  # Show the welcome page with login/register options
    else:
        show_dashboard()

if __name__ == "__main__":
    main()