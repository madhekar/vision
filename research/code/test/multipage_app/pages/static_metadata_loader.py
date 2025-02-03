import streamlit as st
from utils.static_metadata_load_util import static_metadataload as sml

st.subheader("STATIC METADATA: LOAD", divider="gray")

c1, c2 = st.columns([.5,1], gap='medium')

ld = st.button("clean & load")
if ld:
  sml.execute()