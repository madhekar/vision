
import streamlit as st
import time

if "items" not in st.session_state:
    st.session_state.items = range(50)
if "i" not in st.session_state:
    st.session_state.i = 0
if "run" not in st.session_state:
    st.session_state.run = False

def run():
    st.session_state.run = True

@st.fragment()
def frag_writer(container):
    st.button("Start", on_click=run)
    st.button("Stop", on_click=st.stop)
    if st.session_state.run and st.session_state.i < len(st.session_state["items"]):
        container.write(st.session_state["items"][st.session_state.i])
        st.session_state.i += 1
        time.sleep(.2)
        st.rerun(scope="fragment")
    else:
        st.session_state.run = False

header = st.container()
body = st.container()
with header:
    frag_writer(body)