import streamlit as st
from streamlit_extras.app_logo import add_logo
import torch
from zmedia.utils.config_util import config 
from zmedia.utils.util import storage_stat as ss
from zmedia.utils.util import setup_app as sa
import os
import sys
import runpy
from pathlib import Path
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

BASE_FOLDER = Path(__file__).resolve().parent


"""
            data_root,
            app_root,
            zmedia_path,
            zmedia_file,
            zmedia_dest
"""
def load_app_configuration():
    root_data, root_app, current_os = config.app_config_load()
    print(f'app root: {root_app} data root: {root_data}')
    if not os.path.exists(root_data):
        ap, dp, mp,  = config.setup_config_load()
        #sa.folder_setup(ap, dp, mp)
    else:    
        cnt = ss.remove_all_files_by_type(root_app, 'I')
        if cnt > 0:
           print(f' {cnt} : number of files removed')
           #return root_data, root_app
    return current_os


def load_css(css_path):
    with open(file=css_path) as f:
        s = f"<style>{f.read()}</style>"
        st.html(s)

def main():        

    css_path = os.path.join(BASE_FOLDER, "assets", "styles.css")
    print(f'base folder: {BASE_FOLDER} css path: {css_path}')
    load_css(css_path)

    # ar,dr = 
    current_os = load_app_configuration()
    print(f"Current Environment: {current_os}")
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
    
    multimodal_search = st.Page(
        page="pages/multimodal_search.py",
        title="🔎 SEARCH",
        # icon=":material/search:",
    )
    #add_logo("./assets/zmedia_logo.png", height=200)
    st.logo(os.path.join(BASE_FOLDER,"assets","zm_logo_2.png"), size="large") #zm/assets/zm_logo-Picsart-BackgroundRemover.png
    
    # if current_os == "LINUX":

    #     pg = st.navigation(
    #         {
    #             "OVERVIEW": [overview],
    #             "SEARCH": [multimodal_search],
    #         }
    #     )
    # else:
    pg = st.navigation(
            {
                "OVERVIEW": [overview],
                "SEARCH": [multimodal_search],
            }
        )    
    pg.run()

def run_app():
    script_path = os.path.abspath(__file__)
    sys.argv = ['streamlit', 'run', script_path] + sys.argv[1:]
    runpy.run_module('streamlit', run_name='__main__')

if __name__ == "__main__":
    main()