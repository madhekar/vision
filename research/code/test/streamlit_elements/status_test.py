import random
import streamlit as st

st.write(
    """
# Biopsias classification demo
"""
)

print(st.session_state)

text_area = st.empty()

text = text_area.text_area("Text to analyze", "Write a number")

button1 = st.button("Random")
button2 = st.button("Run")

if button1:
    text = text_area.text_area("Text to analyze", str(random.random()))

if button2:
    st.write(text)
