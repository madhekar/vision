import streamlit as st
from utils.preprocess_util import load_vectordb as lvd

st.header("METADATA: LOAD", divider="gray")
lvd.execute()

    