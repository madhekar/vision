import streamlit as st
from utils.static_metadata_load_util import static_metadataload as sml

st.header("STATIC METADATA & MODELS: CREATE", divider="gray")

sml.execute()