import streamlit as st

# Custom CSS to rotate the header text by 90 degrees
vertical_text_style = """
<style>
.vertical-text {
    /* Rotate the text 90 degrees clockwise */
    transform: rotate(90deg);
    /* Adjust transform origin for better positioning, if needed */
    transform-origin: left top;
    /* Add some padding or margin to prevent overlap with surrounding content */
    margin-top: 10px; /* Adjust this value as needed */
    margin-bottom: 10px;
    /* Optional: Style the font/size */
    font-size: .5rem;
    font-weight: bold;
}
</style>
"""

st.markdown(vertical_text_style, unsafe_allow_html=True)

# Use st.markdown with a custom class to apply the style
st.markdown('<p class="vertical-text">This is vertical text</p>', unsafe_allow_html=True)

st.write("---") # Add a horizontal line for separation

st.write("Normal text flows horizontally alongside the vertical text, though layout adjustments with `st.columns` may be needed for precise placement.")
