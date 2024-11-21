import streamlit as st
from utils.datatrim_util import datatrim as dlu

st.subheader("DATA: TRIM (VERIFY)", divider='gray')
dlu.execute()
