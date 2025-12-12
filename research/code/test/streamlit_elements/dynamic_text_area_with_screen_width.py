import streamlit as st

st.set_page_config(layout="wide") # Step 1

# Step 2: Define custom CSS with media queries
custom_css = """
<style>
/* Base styles for the textarea content and the container */
.stTextArea textarea {
    font-size: 16px; /* Default font size */
    height: 150px;  /* Default height */
}

/* Styles for screens smaller than 768px (e.g., mobile devices) */
@media (max-width: 768px) {
    .stTextArea textarea {
        font-size: 14px;
        height: 100px;
    }
}

/* Styles for screens larger than 1200px (e.g., large monitors) */
@media (min-width: 1200px) {
    .stTextArea textarea {
        font-size: 20px;
        height: 250px;
    }
}

/* Optional: Make the overall widget container expand to parent width */
.element-container:has(>.stTextArea), .stTextArea {
    width: 100% !important; 
}
</style>
"""

# Step 3: Inject the CSS
st.markdown(custom_css, unsafe_allow_html=True)

# Display the text area
st.title("Responsive Text Area Example")
user_input = st.text_area("Type your text here and resize your browser window:", 
                          value="Watch how the font size and height change as you resize the browser window.", 
                          height=150) # The height parameter can be set as a default, but CSS will override based on screen size
