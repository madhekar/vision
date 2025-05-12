
import streamlit as st
from utils.preprocess_util import preprocess as pp
from utils.face_util import base_face_train as bft_train
from multipage_app.utils.face_util import base_face_predict as bft_predict
from utils.util import storage_stat as ss
from utils.config_util import config

raw_data_path, input_data_path, app_data_path, final_data_path = (
        config.overview_config_load()
    )

user_source_selected = st.sidebar.selectbox(
        "data source folder",
        options=ss.extract_user_raw_data_folders(raw_data_path),
        label_visibility="collapsed"
    )
st.subheader("METADATA: GENERATE", divider="gray")

c1,c2 = st.columns([1,1])
with c1:
   c11,c12 = st.columns([1,1], gap="small") 
   with c11:
      btn_face = st.button(label='Face Model Generate/ Refresh')
   with c12:   
      btn_gennerate = st.button(label='Generate names in images')
   c1a_status = st.status('refresh people detection model', state='running', expanded=True)
   with c1a_status:
      if btn_face:
          c1a_status.info("starting to create face model.")
          st.info('step 1: train know faces for search...')
          bft.exec(user_source_selected)
          c1a_status.update(label="face detection model complete!", state="complete", expanded=False) 
   c2b_status = st.status('create names from images', state='running', expanded=True)      
   with c2b_status:
      if btn_face:
          c2b_status.info("starting to create names for images using face model.")
          st.info("step 1: detect faces form images...")
          bft.exec(user_source_selected)
          c2b_status.update(
              label="names of people generation from model complete!", state="complete", expanded=False
          )


with c2: 
   btn_metatdata = st.button(label='Metadata Generate')
   c2_status = st.status('create static location store', state='running', expanded=True)
   with c2_status:
      if btn_metatdata:
          st.info('step 2: create metadata for search such as annotations location, text, person names etc...')
          pp.execute(user_source_selected)

   