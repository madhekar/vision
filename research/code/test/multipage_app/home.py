import streamlit as st
import os
import sys
sys.path.append('..')

########################################################################
# https://docs.streamlit.io/get-started/tutorials/create-a-multipage-app
# https://www.youtube.com/watch?v=jbJpAdGlKVY
# presentations - prezi.com
#########################################################################

st.set_page_config(
    page_title="zesha: Media Portal (MP)",
    page_icon="/home/madhekar/work/zsource/zesha-high-resolution-logo.jpeg",
    initial_sidebar_state="expanded",
    layout="wide",
    menu_items={
        'About': 'Zesha PC is created by Bhalchandra Madhekar',
        'Get Help':'https://www.linkedin.com/in/bmadhekar'
    }
)

def load_css(css_path):
    with open(file=css_path) as f:
        s = f"<style>{f.read()}</style>"
        st.html(s)

css_path = os.path.join("assets", "styles.css")

load_css(css_path)

sys.dont_write_bytecode = True

overview = st.Page( 
    page="pages/overview.py", 
    title="OVERVIEW", 
    icon=":material/house:", 
    default=True

)
data_extadd = st.Page(
    page="pages/data_extadd.py",
    title="ADD",
    icon=":material/group_work:",
)

data_trim = st.Page(
    page="pages/data_trim.py",
    title="TRIM",
    icon=":material/group_work:",
)

data_validate = st.Page(
    page='pages/data_validation.py',
    title="VALIDATE",
    icon=":material/group_work:",
)

data_correction = st.Page(
    page='pages/metadata_correction.py',
    title="EDIT",
    icon=":material/edit:"
)

metadata_creater = st.Page(
    page="pages/metadata_creater.py",
    title="GENERATE",
    icon=":material/engineering:",
)

metadata_loader = st.Page(
    page="pages/metadata_loader.py",
    title="LOAD",
    icon=":material/published_with_changes:",
)

static_metadata_loader = st.Page(
    page="pages/static_metadata_loader.py",
    title="CREATE",
    icon=":material/published_with_changes:",
)

multimodal_search = st.Page(
    page="pages/multimodal_search.py",
    title="SEARCH",
    icon=":material/search:",
)

# st.logo("assets/zesha-high-resolution-logo.jpeg")

pg = st.navigation(
    {
        "OVERVIEW": [overview],
        "DATA": [data_extadd, data_trim, data_validate],      
        "METADATA: STATIC": [static_metadata_loader],
        "METADATA: DYNAMIC": [data_correction, metadata_creater, metadata_loader],
        "SEARCH": [multimodal_search],
    }
)

pg.run()