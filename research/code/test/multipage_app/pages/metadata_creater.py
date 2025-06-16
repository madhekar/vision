
import streamlit as st
from utils.preprocess_util import preprocess as pp
from utils.face_util import base_face_train as bft_train
from utils.face_util import base_face_predict as bft_predict
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

# c1,c2 = st.columns([.1,.9], gap="medium", vertical_alignment="top")
# with c1:
#    c = st.container(border=True)
#    btn_face = c.button(label='Refresh: Face Model')
#    c1_status = c.status('refresh people detection model', state='running', expanded=True)
#    with c1_status:
#       if btn_face:
#           c1_status.info("starting to create face model.")
#           st.info('step: - 1: train know faces for search...')
#           bft_train.exec()
#           c1_status.update(label="face detection model complete!", state="complete", expanded=False) 
# with c2: 
#   c_ = st.container(border=True)
status = st.status('refresh people detection model', state='running', expanded=True)
btn_metatdata = st.button(label='Metadata Generate')
with status:
   if btn_metatdata:
        st.info('step 2: create metadata for search such as annotations location, text, person names etc...')
        pp.execute(user_source_selected)

   