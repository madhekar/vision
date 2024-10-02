import streamlit as st

st.title('the callback function always outputs the CURRENT string...')

def callback():
    st.text(st.session_state.mytext)

st.text_input('enter text',value='',on_change=callback,key='mytext')
