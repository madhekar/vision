import os
import getpass
# from utils.config_util import config
import streamlit as st   


def get_user():
    return getpass.getuser()

def get_external_devices(user):
    return os.listdir(f'/media/{user}')


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
       st.sidebar.selectbox(label="Select Source", options=source_list)

if __name__ == "__main__":
    execute()