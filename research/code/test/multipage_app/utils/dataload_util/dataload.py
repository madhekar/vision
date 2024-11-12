import os
import getpass
# from utils.config_util import config
import streamlit as st   


def get_user():
    return getpass.getuser()

def get_external_devices(user):
    return os.listdir(f'/media/{user}')

def display_tree(path, indent=""):
    """Displays the directory tree structure."""
    sout = ""

    for entry in os.listdir(path):
      if os.access(os.path.join(path, entry), os.R_OK):  
        full_path = os.path.join(path, entry)
        if os.path.isdir(full_path):
            sout += indent + "├── " + entry + '\n'
            display_tree(full_path, indent + "|   ")
        else:
            sout += indent + "└── " + entry + '\n'
    return sout        


def execute():
    # (
    #     raw_data_path,
    #     input_image_path,
    #     input_video_path,
    #     input_txt_path,
    # ) = config.dataload_config_load()

    source_list = []
    source_list = get_external_devices(get_user())
    if len(source_list) > 0:
       ext = st.sidebar.selectbox(label="Select Source", options=source_list)
    

    c1, c2 = st.columns([1,1])
    c1.text_area(label="External Source Structure", value= display_tree(os.path.join('/media/madhekar/' , ext)))

if __name__ == "__main__":
    execute()