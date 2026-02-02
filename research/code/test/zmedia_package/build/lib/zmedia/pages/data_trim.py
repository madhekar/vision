import streamlit as st
from utils.datatrim_util import datatrim as dlu

st.header("DATA: TRIM (VERIFY)", divider='gray')

dlu.execute()
