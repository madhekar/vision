import os
import pandas as pd
import streamlit as st
from math import ceil
import pandas as pd
import os
import folium as fl
from streamlit_folium import st_folium
from utils.util import init_streamlit as sti

from utils.config_util import config


# initialize streamlit container UI settings
sti.initUI()

def orchestrator(raw_folders):
   # define application sidebar
   with st.sidebar:
        st.markdown("<p class='big-font-header'>@Home Media Portal</p>", unsafe_allow_html=True)

        st.divider()

        s = st.sidebar.selectbox(label="## Search Modality", options=("text", "image"), index=1, help='select search modality type')

        st.divider()

        ms = st.sidebar.multiselect(
            label="## Result Modalities",
            options=raw_folders,
            help='select one or more search result modalities'
        )

        st.divider()

        return ms

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

# get immediate chield folders 
def extract_user_raw_data_folders(pth):
   return next(os.walk(pth))[1]

def execute():
    rwp, ddp, qdp, mdp,mfp, smfp, vdbp = config.data_validation_config_load()

    raw_folders = extract_user_raw_data_folders(rwp)

    dfo = build_orch_structure(raw_folders)

    orchestrator(raw_folders)

if __name__ == '__main__':
    execute()
   
  