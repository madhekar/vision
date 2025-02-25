import streamlit as st
from utils.preprocess_util import pp1 #preprocess as pp

st.subheader("METADATA: GENERATE", divider="gray")

bcreate_metadata = st.button('create image metadata')
if bcreate_metadata:
   pp1.execute()