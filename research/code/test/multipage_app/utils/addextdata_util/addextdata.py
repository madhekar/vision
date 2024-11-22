
import os
import getpass
import streamlit as st
from utils.util import adddata_util as adu
from streamlit_tree_select import tree_select
from utils.config_util import config

nodes=[]
nodes.append(adu.path_dict("/home/madhekar/work/home-media-app/data/raw-data"))

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

st.sidebar.button(label="Trim(Checked-Folders)")

def get_user():
    return getpass.getuser()


def get_external_devices(user):
    return os.listdir(f"/media/{user}")

def execute():
        (
        raw_data_path,
        input_image_path,
        input_video_path,
        input_txt_path,
    ) = config.dataload_config_load()

    # select load data from external data source such as USB device

    # create new path string by appending raw-data path with source-name from external device

    # check new path raw-data/source-name exists on hard drive
    
    # check folders and files exists in raw-data/source-name folder   

    # remove all folders and files on raw-data/source-name  --- todo think about UNDO requirments

    # copy all folders and files to raw-data/source-name folder
        
    # arc_folder_name = util.get_foldername_by_datetime()
    # archive_dup_path = os.path.join(archive_dup_path, arc_folder_name)
    # dr = DuplicateRemover(dirname=input_image_path, archivedir=archive_dup_path)
    # dr.find_duplicates()


if __name__ == "__main__":
    execute()