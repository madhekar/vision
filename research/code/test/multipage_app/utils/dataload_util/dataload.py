import os
import getpass
from utils.config_util import config
import streamlit as st   
import util
from utils.util import adddata_util as adu
from streamlit_tree_select import tree_select


def get_user():
    return getpass.getuser()

def get_external_devices(user):
    return os.listdir(f'/media/{user}')

def get_path_as_dict(path):
    nodes = []
    nodes.append(adu.path_dict(path))
    return nodes

def display_folder_tree(nodes):
    con = st.sidebar.container(height=500)
    with con:
        return_select = tree_select(nodes, no_cascade=True)

    con1 = st.sidebar.expander(label="checked folders")  # container(height=500)
    with con1:
        selected = []
        for e in return_select["checked"]:
            e0 = e.split("@@")[0]
            selected.append(e0)
            st.write(e0)

    


def execute():
    (
        raw_data_path,
        input_image_path,
        input_video_path,
        input_txt_path,
    ) = config.dataload_config_load()

    display_folder_tree(get_path_as_dict(raw_data_path))
    st.sidebar.button(label="Trim(Checked-Folders)")  
    
    # source_list = []
    # source_list = get_external_devices(get_user())
    # if len(source_list) > 0:
    #    ext = st.sidebar.selectbox(label="Select Source", options=source_list)

    # c1.text_area(label="External Source Structure", value= display_tree(os.path.join('/media/madhekar/' , ext)))

if __name__ == "__main__":
    execute()