import streamlit as st
from streamlit_extras.stylable_container import stylable_container

st.markdown("### Default text_input height")
st.text_input("Normal Input")

st.markdown("### text_input with custom height via CSS (smaller font/padding)")

with stylable_container(
    key="custom_input_height",
    css_styles="""
        div[data-baseweb="input"] > input {
            padding-top: 0px;
            padding-bottom: 0px;
            min-height: 8px;
            height: 8px; /* Adjust as needed */
            font-size: 7px; /* Smaller font size contributes to a smaller height */
        }
    """,
):
    st.text_input("Smaller Input")

st.markdown("### Using st.text_area for multi-line input")
st.text_area("Text Area (can set height directly)", height=50) # height parameter works for text_area
