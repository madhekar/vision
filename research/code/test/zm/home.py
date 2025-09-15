import streamlit as st
from utils.config_util import config 
from utils.util import storage_stat as ss
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
    page_icon="../assets/zesha-high-resolution-logo.jpeg",  #check
    initial_sidebar_state="expanded",
    layout="wide",
    menu_items={
        'About': 'Zesha PC is created by Bhalchandra Madhekar',
        'Get Help':'https://www.linkedin.com/in/bmadhekar'
    }
)
def load_app_configuration():
    root_data, root_app = config.app_config_load()
    print(f'app root: {root_app} data root: {root_data}')
    cnt = ss.remove_all_files_by_type(root_app, 'I')
    print(f' {cnt} : number of files removed')
    #return root_data, root_app

def load_css(css_path):
    with open(file=css_path) as f:
        s = f"<style>{f.read()}</style>"
        st.html(s)

css_path = os.path.join("assets", "styles.css")
load_css(css_path)

# ar,dr = 
load_app_configuration()
# if 'app_root' not in st.session_state:
#     st.session_state['app_root'] = ar

# if 'data_root' not in st.session_state:
#     st.session_state['data_root'] = dr

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

data_validation = st.Page(
    page='pages/validate.py',
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
        "DATA": [data_extadd, data_trim, data_validation],      
        "METADATA: STATIC": [static_metadata_loader],
        "METADATA: DYNAMIC": [data_correction, metadata_creater, metadata_loader],
        "SEARCH": [multimodal_search],
    }
)

pg.run()