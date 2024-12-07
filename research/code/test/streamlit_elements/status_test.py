import random
import streamlit as st

st.write(
"""
# Biopsias classification demo
"""
)

print(st.session_state)

ta = st.empty()

text = ta.text_area("Text to analyze", "Write a number")

button1 = st.button("Random")
button2 = st.button("Run")

if button1:
    text = ta.text_area("Text to analyze", str(random.random()))

if button2:
    st.write(text)
