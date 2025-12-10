import streamlit as st
from streamlit_extras.stylable_container import stylable_container

with stylable_container(
    key="button_container",
    css_styles="""
        button {
            height: 2rem;  /* Set the desired height */
            width: 8rem;  /* Set the desired width */
            font-size: .4rem; /* Optional: adjust font size for better fit */
        }
    """,
):
    st.button("Click Me!")