import streamlit as st
from st_click_detector import click_detector

content ="""
    <p><a href='#' id='Link 1'>First link</a></p>
    <a href='#' id='Image 1'><img width='20%' src='https://images.unsplash.com/photo-1565130838609-c3a86655db61?w=200'></a>
    """
clicked = click_detector(content)

st.markdown(f"**{clicked} clicked**" if clicked != "" else "**No click**")
