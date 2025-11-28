import streamlit as st

# Function to load CSS file
def apply_custom_css(css_file):
    with open(css_file) as f:
        css = f.read()
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# Load the external CSS file
apply_custom_css('style.css') # Make sure you have a style.css file in the same directory

# Define multiple text inputs with unique labels (used as aria-label in HTML)
st.text_input("First Name", key="first_name")
st.text_input("Last Name", key="last_name")
st.text_input("Address", key="address")
