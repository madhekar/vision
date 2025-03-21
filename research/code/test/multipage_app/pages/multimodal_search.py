import streamlit as st
from utils.search_util import search as s

st.subheader("MULTI-MODAL: SEARCH", divider="gray")

s.execute()