import os
import streamlit as st


st.title("Overview")

def do_work():
  st.write("do work")
  st.button("Metadata Done", key="green")


v = st.button("Metadata Orchestration", key="gray", on_click=do_work())

  

#st.button("command", key="green")

st.button("command", key="orange")

st.button("Metadata Orchestration", key="pulse")