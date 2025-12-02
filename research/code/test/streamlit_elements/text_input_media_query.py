import streamlit as st

st.set_page_config(layout="wide") # Optional: start with wide layout

# Custom CSS with a media query
custom_css = """
<style>
/* Default styles for the text input (optional) */
div[data-testid="stTextInput"] input[type="text"] {
    width: 100%;
    max-width: 500px; /* Default max-width for larger screens */
    margin: 0 auto; /* Center the input if it's narrower than its container */
}

/* Media query for screens with a maximum width of 768px (e.g., mobile devices) */
@media (max-width: 1768px) {
    div[data-testid="stTextInput"] input[type="text"] {
        size: 5px;
        max-width: 90vw; /* Set a different max-width for smaller screens (e.g., 90% of viewport width) */
    }
}
</style>
"""

# Inject the custom CSS into the Streamlit app
st.markdown(custom_css, unsafe_allow_html=True)

st.title("CSS Media Query on st.text_input")

# Place the text input component
user_input = st.text_input("Enter some text:")

st.write(f"You entered: {user_input}")

# Alternative method to adjust width for specific inputs using columns (without media query)
st.markdown("---")
st.write("Using `st.columns` for static width control:")
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    short_input = st.text_input("Shorter Input:")
