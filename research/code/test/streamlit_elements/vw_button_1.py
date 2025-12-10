import streamlit as st

# Define your custom CSS with viewport units
custom_css = """
<style>
    /* Target a specific Streamlit element by its generated class name */
    /* You can inspect elements in your browser's developer tools to find these */
    .stButton > button {
        background-color: #4CAF50; /* Example background color */
        color: white; /* Example text color */
        padding: 1vw 2vw; /* Padding using viewport width units */
        font-size: 1.5vw; /* Font size using viewport width units */
        border-radius: 0.5vw; /* Border radius using viewport width units */
        border: none;
        cursor: pointer;
    }

    /* Style for a custom div element */
    .my-custom-div {
        background-color: #f0f0f0;
        padding: 2vw;
        margin-bottom: 1vw;
        border-radius: 1vw;
        font-size: 1.2vw;
    }

    /* General styling for the Streamlit app */
    body {
        font-family: Arial, sans-serif;
    }
</style>
"""

# Inject the custom CSS into the Streamlit app
st.markdown(custom_css, unsafe_allow_html=True)

st.subheader("Custom CSS with Streamlit Elements")

# Example using a Streamlit button with custom CSS
if st.button("Click Me!"):
    st.write("Button clicked!")

# Example using a custom div element with custom CSS
st.markdown('<div class="my-custom-div">This is a custom div with specific styling.</div>', unsafe_allow_html=True)

st.write("This is some regular Streamlit text.")