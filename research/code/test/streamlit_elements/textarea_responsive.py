import streamlit as st

st.markdown("""
<style>
/* Targets the specific textarea element within the Streamlit structure */
div[data-baseweb="base-input"] > textarea {
    min-height: 9px; /* Set your desired min-height here */
    font-size: 8px;
    padding: 10px; /* Adjust padding as needed */
}
</style>
""", unsafe_allow_html=True)

# The text area will now respect the custom CSS min-height
user_input = st.text_area("Enter text here", height=100)