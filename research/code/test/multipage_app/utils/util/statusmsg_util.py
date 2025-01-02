import streamlit as st
from collections import defaultdict

def init():
    if "btn_prsd_status" not in st.session_state:
        st.session_state.btn_prsd_status = [0] * 4

    if "msgs" not in st.session_state:
        st.session_state.msgs = {"load": [], "duplicate": [], "quality": [], "metadata": []}


def add_messages(msg_type, message):
    st.session_state.msgs[msg_type].append(message)

def get_message_by_type(tmsg):
    d = defaultdict(set)
    for ele in st.session_state.msgs[tmsg]:
        k, v =  ele.split("|")
        d[k].add(v)
    return d

