import os
import streamlit as st
from utils.face_util import base_face_infer as bftf
from utils.config_util import config

def predict_names(ibtf, img):
    names = ibtf.predict_names(img)
    return names

@st.cache_resource
def init():
    faces_dir,  class_embeddings_folder, class_embeddings, label_encoder_path, label_encoder, faces_svc_path, faces_svc = config.faces_config_load()

    faces_embeddings, faces_label_enc, faces_model_svc = (
        os.path.join(class_embeddings_folder, class_embeddings),
        os.path.join(label_encoder_path,label_encoder),
        os.path.join(faces_svc_path,faces_svc)
    )
    ibtf = bftf.infer_faces(faces_embeddings, faces_label_enc, faces_model_svc)    
    print(f'-->init face test{label_encoder_path}{faces_svc_path}')
    return ibtf

def exec():
    img_path = '/home/madhekar/work/home-media-app/data/input-data/img'
    ibtf = init()
    for img in os.listdir(img_path):
      nfaces, names = predict_names(ibtf, os.path.join(img_path, img))
      print(f' file: {img} # of faces: {nfaces} identified faces: {names}')

# kick-off face training generation
if __name__ == "__main__":
    exec()
    