import streamlit as st
from utils.search_util import search as s

st.subheader("SEARCH", divider="gray")

s.execute()