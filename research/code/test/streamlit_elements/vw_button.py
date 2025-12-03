import streamlit as st
from streamlit_extras.stylable_container import stylable_container

with stylable_container(
    key="custom_button_container",
    css_styles="""
        button {
            width: 30vw; /* 30% of viewport width */
            font-size: 2vw;
            background-color: orange;
            border-radius: 1vw;
        }
        button:hover {
            background-color: darkorange;
        }
    """,
):
    st.button("Click Me (30vw Wide)")

if st.button("Default Streamlit Button"):
    st.write("This button has the default style.")
