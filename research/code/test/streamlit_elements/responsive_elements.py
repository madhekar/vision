import streamlit as st

# 1. Add your elements with unique keys
st.text_input("Enter text", key="my_textbox")
st.text_input("Age:", key="age_textbox")
st.button("Click Me", key="my_button")
st.selectbox("Choose an option:", ["Option 1", "Option 2"], key="my_dropdown")

# 2. Inject custom CSS targeting the specific keys
st.markdown("""
<style>
    /* Target the container of 'my_textbox' */
    .st-key-my_textbox input {
        background-color: #f0f8ff; /* Change textbox background */
        border: 2px solid #4CAF50;
        border-radius: 5px;
    }
            
    .st-key-age_textbox input { background-color: #ff0000;}        

    /* Target the container of 'my_button' */
    .st-key-my_button button {
        background-color: #4CAF50; /* Change button color */
        color: white;
        padding: 10px 24px;
        cursor: pointer;
        width: 50%; /* Make the button responsive to its column width */
    }

    /* Target the container of 'my_dropdown' */
    .st-key-my_dropdown div[data-baseweb="select"] {
        background-color: #ffffe0; /* Change dropdown background */
    }

    /* Use CSS Media Queries for responsiveness in styling (e.g., small screens) */
    @media (max-width: 600px) {
        .st-key-my_textbox input, .st-key-age_textbox input, .st-key-my_button button, .st-key-my_dropdown div[data-baseweb="select"] {
            font-size: 8px;
        }
    }
</style>
""", unsafe_allow_html=True)
