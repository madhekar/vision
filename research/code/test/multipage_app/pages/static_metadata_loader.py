import streamlit as st
from utils.static_metadata_load_util import static_metadataload as sml

st.subheader("STATIC METADATA: LOAD", divider="gray")

sml.execute()