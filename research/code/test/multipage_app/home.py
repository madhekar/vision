import streamlit as st

data_orchestration = st.Page(
    page='pages/data_orchestration.py',
    title="Data orchistration Workflow",
    icon=":material/smart_toy:",
    default=True
)

data_correction = st.Page(
    page='pages/metadata_fix.py',
    title="Data ofix Workflow",
    icon=":material/smart_toy:"
)

pg = st.navigation(
    {
    "workflow": [data_orchestration],
    "correction": [data_correction]
    }
    )

st.logo("assets/zesha-high-resolution-logo.jpeg")
st.sidebar.text("At Home Media Portal")

pg.run()