import os
import getpass
from utils.config_util import config
import streamlit as st   
import util
from utils.util import adddata_util as adu
from streamlit_tree_select import tree_select
from utils.util import model_util as mu
from utils.util import storage_stat as ss

def get_path_as_dict(path):
    nodes = []
    nodes.append(adu.path_dict(path))
    return nodes

#@st.fragment
def display_folder_tree(nodes):
    con = st.sidebar.container(height=500, border=False)
    with con:
        return_select = tree_select(nodes, no_cascade=True) 
    display_folder_stats(return_select["checked"])

    exp = st.sidebar.expander(label="CHECKED FOLDERS TO TRIM")  # container(height=500)
    with exp:

        for e in return_select["checked"]:
            e0 = e.split("@@")[0]
            st.write(e0)


def display_folder_stats(flist):
    row_size = 4
    page =1 
    batch_size = 12
    batch = flist[(page - 1) * batch_size : page * batch_size]
    grid = st.columns(row_size, gap="small", vertical_alignment="top")
    col = 0
    st.cache_resource
    for df in batch:
        with grid[col]:
            folder = df.split("@@")[0]
            dfolder = folder.split("/")[-1]
            st.subheader(dfolder, divider='gray')
            st.bar_chart(
                ss.extract_folder_stats(folder)['count'],
                stack=True,
                horizontal=False,
                y_label="total file count per filetype",
                color=['#768A76']
            )
            st.bar_chart(
                ss.extract_folder_stats(folder)["size"],
                stack=True,
                horizontal=False,
                y_label="total file size per filetype",
                color=["#8CBD8C"]
            )

        col = (col + 1) % row_size
    
def execute():
    (
        raw_data_path,
        input_image_path,
        input_video_path,
        input_txt_path,
    ) = config.dataload_config_load()


    '''
    select data source to trim data
    ''' 
    source_list = []
    #source_list = get_external_devices(get_user())
    source_list = mu.extract_user_raw_data_folders(raw_data_path)
    if len(source_list) > 0:
       ext = st.sidebar.selectbox(label="Select Source", options=source_list)


    st.sidebar.caption("CHECK FOLDERS TO TRIM",unsafe_allow_html=True)
    display_folder_tree(get_path_as_dict(os.path.join(raw_data_path, ext)))
    st.sidebar.button(label="TRIM CHECKED FOLDERS",use_container_width=True) 
    # c1.text_area(label="External Source Structure", value= display_tree(os.path.join('/media/madhekar/' , ext)))



if __name__ == "__main__":
    execute()