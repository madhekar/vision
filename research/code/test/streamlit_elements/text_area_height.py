import streamlit as st

col1, col2 = st.columns([1, 3])

st.markdown(
"""
<style>
div[data-baseweb="base-input"] > textarea {
    min-height: 4px;
    font-size: 8px; !important;
    padding: 0;
}
</style>
""", unsafe_allow_html=True
)

height = st.slider("Set the height of the text area", 1, 1000, 10)

with col1:
    st.header("Instructions")

with col2:
    col21, col22 = st.columns(2)
with col21:
    Product_Name = st.text_area("Provide the product name", height=height)
with col22:
    Campaign_Name = st.text_area("Provide the name of campaign", height=height)
