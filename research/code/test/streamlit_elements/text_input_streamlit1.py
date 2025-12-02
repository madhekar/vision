import streamlit as st

custom_css = """
<style>
/* This selector might need adjustment based on Streamlit version and element structure */
div[data-baseweb="input"] > div > input {
    width: 500px !important; /* Adjust width as needed */
    height: 20px !important;
    font-size: 120% !important;
}
</style>
"""
custom_css_1 = """
<style>
div[data-baseweb="base-input"]{ 
background:linear-gradient(to bottom, #3399ff 0%,#00ffff 100%);
height: 20px;
width: 100px;
border: 2px;
border-radius: 3px;
}

input[class]{
font-weight: bold;
font-size:60%;
color: white;
}
</style>
"""
st.markdown(custom_css_1, unsafe_allow_html=True)
st.title("Custom Text Input Width")
user_input = st.text_input("Enter text:")