import streamlit as st


def init():
    #mystate = st.session_state
    if "btn_prsd_status" not in st.session_state:
        st.session_state.btn_prsd_status = [0] * 4

    if "msgs" not in st.session_state:
        st.session_state.msgs = {"load": [], "duplicate": [], "quality": [], "metadata": []}


def add_messages(msg_type, message):
    st.session_stat.msgs[msg_type].append(message)

def get_message_by_type(tmsg):
    return st.session_state.msgs[tmsg]
