
import util
import os
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


def execute():
    input_image_path, archive_dup_path = config.dedup_config_load()
    arc_folder_name = util.get_foldername_by_datetime()
    archive_dup_path = os.path.join(archive_dup_path, arc_folder_name)
    dr = DuplicateRemover(dirname=input_image_path, archivedir=archive_dup_path)
    dr.find_duplicates()


if __name__ == "__main__":
    execute()