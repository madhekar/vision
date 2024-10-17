import streamlit as st


st.title("At Home Media Portal")

data_orchestration = st.Page(
    page='pages/data_orchestration.py',
    title="Data orchistration Workflow",
    icon=":material/smart_toy:"
)

data_correction = st.Page(
    page='pages/metadata_fix.py',
    title="Data ofix Workflow",
    icon=":material/smart_toy:"
)