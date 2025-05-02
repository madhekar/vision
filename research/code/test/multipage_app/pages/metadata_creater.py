import streamlit as st
from utils.preprocess_util import preprocess as pp
from utils.face_util import base_face_train as bft

st.subheader("METADATA: GENERATE", divider="gray")

c1,c2 = st.columns([1,1])
with c1:
   btn_face = st.button(label='Face Model Gen')
   if btn_face:
     st.info('step 1: train know faces for search...')
     bft.exec()

with c2: 
   btn_metatdata = st.button(label='Metadata Gen')
   if btn_metatdata:
     st.info('step 2: create metadata for search such as annotations location, text, person names etc...')
     pp.execute()

   