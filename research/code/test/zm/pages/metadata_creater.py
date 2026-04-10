
import streamlit as st
from utils.preprocess_util import preprocess as pp
from utils.preprocess_util import preprocess_video as ppv
from utils.util import storage_stat as ss
from utils.config_util import config

(raw_data_path, input_data_path, app_data_path, final_data_path,*_) = config.overview_config_load()

user_source_selected = st.sidebar.selectbox("data source folder", options=ss.extract_user_raw_data_folders(raw_data_path), label_visibility="collapsed")

st.header("METADATA: GENERATE", divider="gray")

c1, c2, c3 = st.columns([.1, .1, .8])

with c1:
        btn_metatdata = st.button(label='Image Metadata Generater', type='primary')
        if btn_metatdata:
                st.info('step 2: create metadata for search such as annotations location, text, person names etc...')
                pp.execute(user_source_selected)

with c2:
        btn_metatdata = st.button(label='Video Metadata Generater', type='primary')
        if btn_metatdata:
                st.info('step 2: create metadata for search such as annotations location, text, person names etc...')
                ppv.execute(user_source_selected)


   