
import streamlit as st
from utils.preprocess_util import preprocess as pp
from utils.face_util import base_face_train as bft

st.subheader("METADATA: GENERATE", divider="gray")

c1,c2 = st.columns([1,1])
with c1:
   
   btn_face = st.button(label='Face Model Generate/ Refresh')
   c1_status = st.status('create static location store', state='running', expanded=True)
   with c1_status:
      if btn_face:
          c1.info("starting to create face model.")
          st.info('step 1: train know faces for search...')
          bft.exec()
          c1_status.update(label="metadata creation complete!", state="complete", expanded=False)  


with c2: 
   btn_metatdata = st.button(label='Metadata Generate')
   c2_status = st.status('create static location store', state='running', expanded=True)
   with c2_status:
      if btn_metatdata:
          st.info('step 2: create metadata for search such as annotations location, text, person names etc...')
          pp.execute()

   