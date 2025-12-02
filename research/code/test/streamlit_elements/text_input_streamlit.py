import streamlit as st

st.set_page_config(layout="wide")

st.markdown("""
    <style>
    /* Target the input field within the st.text_input widget */
    div[data-testid="stTextInput"] input[type="text"] {
        background-color: #e6f7ff; /* Light blue background */
        border: 1px solid #007bff; /* Blue border */
        border-radius: 1px;
        padding: 2px;
        color: #333;
        font-size: 4px;
    }

    /* Target the label of the st.text_input widget */
    div[data-testid="stTextInput"] label {
        font-weight: bold;
        color: #007bff;
    }
    </style>
""", unsafe_allow_html=True)

user_input = st.text_input("Enter your name:", "John Doe")

st.write(f"Hello, {user_input}!")
