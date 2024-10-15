import os
import yaml
import pprint
import pandas as pd
import streamlit as st
from math import ceil
import pandas as pd
import os
import folium as fl
from streamlit_folium import st_folium
import init_streamlit as sti



# initialize streamlit container UI settings
sti.initUI()

def orchestrator():
   # define application sidebar
   with st.sidebar:
        st.markdown("<p class='big-font-header'>@Home Media Portal</p>", unsafe_allow_html=True)

        st.divider()

        s = st.selectbox(label="## Search Modality", options=("text", "image"), index=1, help='select search modality type')

        st.divider()

        ms = st.multiselect(
            label="## Result Modalities",
            options=["image", "text", "video", "audio"],
            default=["image", "text"],
            help='select one or more search result modalities'
        )

        st.divider()

st.markdown("<p class='big-font-title'>Data Orchestrator - Home Media Portal</p>", unsafe_allow_html=True)
st.logo("/home/madhekar/work/home-media-app/app/zesha-high-resolution-logo.jpeg")

def build_orch_structure(raw_folders):
    columns = ['duplicate', 'quality', 'missing', 'metadata', 'state']
    data =[]
    idx = []
    for folder in raw_folders:
        data.append([folder + "_dup",folder + "_qua",folder + "_mis", folder + "_met", "0"])
        idx.append(folder)
    df = pd.DataFrame(data=data, columns=columns, index=idx)
    return df
 

def extract_user_raw_data_folders(pth):
   return next(os.walk(pth))[1]

@st.cache_resource
def config_load():
    with open("data_orchestration_conf.yaml") as prop:
        dict = yaml.safe_load(prop)

        pprint.pprint("* * * * * * * * *  * Data Orchestration Properties * * * * * * * * * * *")
        pprint.pprint(dict)
        pprint.pprint("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *")

        raw_data_path = dict["raw-data"]["base_path"]
        duplicate_data_path = dict["duplicate"]["base_path"]
        quality_data_path = dict["quality"]["base_path"]
        missing_data_path = dict["missing"]["base_path"]
        metadata_file_path = dict["metadata"]["base_path"]
        static_metadata_file_path = dict["static-metadata"]["base_path"]
        vectordb_path = dict["vectordb"]["base_path"]

    return (
        raw_data_path,
        duplicate_data_path,
        quality_data_path,
        missing_data_path,
        metadata_file_path,
        static_metadata_file_path,
        vectordb_path
    )




if __name__ == '__main__':
    rwp, ddp, qdp, mdp,mfp, smfp, vdbp = config_load()

    raw_folders = extract_user_raw_data_folders(rwp)

    dfo = build_orch_structure(raw_folders)

    orchestrator()
   
  