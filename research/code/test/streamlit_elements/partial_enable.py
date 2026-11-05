import streamlit as st

# Track or check a value in session state or a first selectbox
box1 = st.selectbox("Choose the first option", ["Enable Box 2", "Disable Box 2"], index=0)

# Disable the second selectbox if 'Disable Box 2' is selected
is_disabled = box1 == "Disable Box 2"

box2 = st.selectbox(
    "Choose the second option",
    ["Option A", "Option B", "Option C"],
    disabled=is_disabled,
)
