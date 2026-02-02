import streamlit as st
from utils.addextdata_util import addextdata as aed

st.header("DATA: EXTERNAL LOAD", divider="gray")
aed.execute()
