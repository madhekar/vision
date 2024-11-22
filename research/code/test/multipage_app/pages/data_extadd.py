import streamlit as st
from utils.addextdata_util import addextdata as aed

st.subheader("DATA: EXTERNAL ADD", divider="gray")
aed.execute()
