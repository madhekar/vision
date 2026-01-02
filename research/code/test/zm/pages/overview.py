import streamlit as st
from utils.overview_util import overview as ovr

st.header("OVERVIEW: DISK STORAGE", divider="gray")
with st.spinner('In Progress...'):
   ovr.execute()

