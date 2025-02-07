import streamlit as st
from utils.preprocess_util import preprocess as pp

st.subheader("METADATA: GENERATE", divider="gray")

pp.execute()