import streamlit as st
import os


# https://docs.streamlit.io/get-started/tutorials/create-a-multipage-app
# https://www.youtube.com/watch?v=jbJpAdGlKVY

#sti.initUI()

st.set_page_config(
    page_title="zesha: Home Media Portal (HMP)",
    page_icon="/home/madhekar/work/zsource/zesha-high-resolution-logo.jpeg",
    initial_sidebar_state="auto",
    layout="wide",
)

def load_css(css_path):
    with open(file=css_path) as f:
        s = f"<style>{f.read()}</style>"
        st.html(s)


css_path = os.path.join("assets", "styles.css")

load_css(css_path)


overview = st.Page( 
    page="pages/overview.py", 
    title="OVERVIEW", 
    icon=":material/house:", 
    default=True
)

data_validation = st.Page(
    page='pages/data_validation.py',
    title="DATA VALIDATE",
    icon=":material/group_work:",
)

data_correction = st.Page(
    page='pages/metadata_correction.py',
    title="CORRECT",
    icon=":material/edit:"
)

metadata_creator = st.Page(
    page="pages/metadata_creator.py",
    title="GENERATE",
    icon=":material/engineering:",
)

metadata_loader = st.Page(
    page="pages/metadata_loader.py",
    title="LOAD",
    icon=":material/published_with_changes:",
)

multimodal_search = st.Page(
    page="pages/multimodal_search.py",
    title="SEARCH",
    icon=":material/search:",
)

pg = st.navigation(
    {
    "OVERVIEW": [overview],
    "DATA: ADD/ VALIDATE": [data_validation],
    "METADATA": [data_correction, metadata_creator, metadata_loader],
    "SEARCH": [multimodal_search]
    }
    )

st.logo("assets/zesha-high-resolution-logo.jpeg")
#st.sidebar.text("Home Media Portal")


pg.run()