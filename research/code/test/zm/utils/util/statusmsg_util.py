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

def show_info_msgs_by_type(tmsg):
    for e in st.session_state.msgs[tmsg]:
        k, v = e.split('|')
        if k == 's':
            st.info(str(v))

def show_warn_msgs_by_type(tmsg):
    for e in st.session_state.msgs[tmsg]:
        k, v = e.split("|")
        if k == "w":
            st.warning(str(v))        

def show_err_msgs_by_type(tmsg):
    for e in st.session_state.msgs[tmsg]:
        k, v = e.split("|")
        if k == "e":
            st.error(str(v))                

def show_all_msgs_by_type(tmsg):
    for e in st.session_state.msgs[tmsg]:
        k, v = e.split("|")
        if k == "e":
            st.error(str(v))
        elif k == 'w':
            st.warning(str(v))    
        else:
            st.info(str(v))    