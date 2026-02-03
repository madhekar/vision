import streamlit as st
from streamlit_extras.app_logo import add_logo
import torch
from utils.config_util import config 
from utils.util import storage_stat as ss
from utils.util import setup_app as sa
import os
import sys
import runpy
torch.classes.__path__ = []
sys.path.append('..')

########################################################################
# https://docs.streamlit.io/get-started/tutorials/create-a-multipage-app
# https://www.youtube.com/watch?v=jbJpAdGlKVY
# presentations - prezi.com
# https://github.com/milvus-io/bootcamp/tree/master/bootcamp/RAG/advanced_rag
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

"""
            data_root,
            app_root,
            zmedia_path,
            zmedia_file,
            zmedia_dest
"""
def load_app_configuration():
    root_data, root_app = config.app_config_load()
    print(f'app root: {root_app} data root: {root_data}')
    if not os.path.exists(root_data):
        ap, dp, mp,  = config.setup_config_load()
        sa.folder_setup(ap, dp, mp)
    else:    
        cnt = ss.remove_all_files_by_type(root_app, 'I')
        if cnt > 0:
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
    title="🏠 OVERVIEW", 
    #icon=":bar_chart:", 
    default=True

)
data_extadd = st.Page(
    page="pages/data_extadd.py",
    title="📥 ADD",
    #icon=":material/group_work:",
)

data_trim = st.Page(
    page="pages/data_trim.py",
    title="✂️ TRIM",
    #icon=":material/group_work:",
)

data_validation = st.Page(
    page="pages/validate.py",
    title="🧐 VALIDATE",
    #icon=":material/group_work:",
)

data_correction = st.Page(
    page='pages/metadata_correction.py',
    title="✍️ EDIT",
    #icon=":material/edit:"
)

metadata_creater = st.Page(
    page="pages/metadata_creater.py",
    title="⚙️ GENERATE",
    # icon=":material/engineering:",
)

metadata_loader = st.Page(
    page="pages/metadata_loader.py",
    title="🛠️ LOAD",
    # icon=":material/published_with_changes:",
)

static_metadata_loader = st.Page(
    page="pages/static_metadata_loader.py",
    title="🪄✨ CREATE",
    # icon=":material/published_with_changes:",
)

multimodal_search = st.Page(
    page="pages/multimodal_search.py",
    title="🔎 SEARCH",
    # icon=":material/search:",
)
#add_logo("./assets/zmedia_logo.png", height=200)
st.logo("assets/zm_logo_2.png", size="large") #zm/assets/zm_logo-Picsart-BackgroundRemover.png

pg = st.navigation(
    {
        "OVERVIEW": [overview],
        "DATA": [data_extadd, data_trim, data_validation],      
        "METADATA: STATIC": [static_metadata_loader],
        "METADATA: DYNAMIC": [data_correction, metadata_creater, metadata_loader],
        "SEARCH": [multimodal_search],
    }
)



if __name__ == "__main__":
    pg.run()