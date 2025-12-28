import streamlit as st
from streamlit_dimensions import st_dimensions

st.set_page_config(layout="wide")

# Get the width of the main content area
width = st_dimensions(key="main_width")
st.write(f"Main container width: {width} pixels")

# Example with columns
col1, col2 = st.columns([0.75, 0.25])
with col1:
    col1_width = st_dimensions(key="col1_width")
    col1_height = st_dimensions(key="col1_height")
    st.write(f"Column 1 width: {col1_width} : {col1_height} pixels")

with col2:
    col2_width = st_dimensions(key="col2_width")
    st.write(f"Column 2 width: {col2_width} pixels")
