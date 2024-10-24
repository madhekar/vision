import streamlit as st
import streamlit_init as sti
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
    title=" Overview", 
    con=":material/house:", 
    default=True
)

storage_initialization = st.Page(
    page="pages/data_storage.py",
    title="Storage Initialization (new)",
    icon=":material/smart_toy:",
)

data_orchestration = st.Page(
    page='pages/data_orchestration.py',
    title="Data Orchestration",
    icon=":material/group_work:",
)

data_correction = st.Page(
    page='pages/metadata_correction.py',
    title="Metadata Correction",
    icon=":material/edit:"
)

metadata_creator = st.Page(
    page="pages/metadata_creator.py",
    title="Metadata Creation",
    icon=":material/engineering:",
)

metadata_loader = st.Page(
    page="pages/metadata_loader.py",
    title="Metadata Loader",
    icon=":material/published_with_changes:",
)

multimodal_search = st.Page(
    page="pages/multimodal_search.py",
    title="multimodal search",
    icon=":material/search:",
)

pg = st.navigation(
    {
    "overview": [overview],
    "storage initialization": [storage_initialization],
    "data orchestration": [data_orchestration],
    "data correction": [data_correction],
    "metadata creator":[metadata_creator],
    "metadata loader": [metadata_loader],
    "multimodal search": [multimodal_search]
    }
    )

st.logo("assets/zesha-high-resolution-logo.jpeg")
#st.sidebar.text("Home Media Portal")


pg.run()