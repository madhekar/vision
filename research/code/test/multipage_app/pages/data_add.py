import streamlit as st
from utils.dataload_util import dataload as dlu

st.subheader("DATA: TRIM (VERIFY)", divider='gray')
dlu.execute()
