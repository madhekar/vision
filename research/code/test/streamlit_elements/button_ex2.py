import streamlit as st
import pandas as pd

if 'name' not in st.session_state:
    st.session_state['name'] = 'Bhalchandra Madhekar'

st.header(st.session_state['name'])

if st.button('Bhal'):
    st.session_state['name'] = 'Bhal Madhekar'
    st.rerun()

if st.button('Bhalchandra'):
    st.session_state['name'] = 'Bhalchandra Madhekar'
    st.rerun()

st.header(st.session_state['name'])
