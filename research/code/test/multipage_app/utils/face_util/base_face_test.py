import os
import streamlit as st
from utils.face_util import base_face_infer as bftf
from utils.config_util import config

class base_face_res:

    def __init__(self):
        self.faces_dir = None,
        self.class_embeddings_folder= None,
        self.class_embeddings= None,
        self.label_encoder_path= None,
        self.label_encoder= None,
        self.faces_svc_path= None,
        self.faces_sv= None,

    async def init(self):
        (
            self.faces_dir ,  
            self.class_embeddings_folder,
            self.class_embeddings,
            self.label_encoder_path,
            self.label_encoder,
            self.faces_svc_path,
            self.faces_sv,
        ) = config.faces_config_load()

        self.faces_embeddings, self.faces_label_enc, self.faces_model_svc = (
            os.path.join(self.class_embeddings_folder, self.class_embeddings),
            os.path.join(self.label_encoder_path, self.label_encoder),
            os.path.join(self.faces_svc_path, self.faces_svc),
        )
        self.faces_infer_obj = await bftf.infer_faces(
            self.faces_embeddings, self.faces_label_enc, self.faces_model_svc
        )

    def pred_names_of_people(self, img):
        _,names =  self.faces_infer_obj.predict_names(img)    
        return ', '.join(names)


# def pred_names(ibtf, img):
#     names = ibtf.predict_names(img)
#     return names

# @st.cache_resource
# def init():
#     (faces_dir,  class_embeddings_folder, class_embeddings, label_encoder_path, label_encoder, faces_svc_path, faces_svc) = config.faces_config_load()

#     faces_embeddings, faces_label_enc, faces_model_svc = (
#         os.path.join(class_embeddings_folder, class_embeddings),
#         os.path.join(label_encoder_path,label_encoder),
#         os.path.join(faces_svc_path,faces_svc)
#         )
#     ibtf = bftf.infer_faces(faces_embeddings, faces_label_enc, faces_model_svc)    
#     print(f'-->init face test{label_encoder_path}{faces_svc_path}')
#     return ibtf

# def exec():
#     img_path = '/home/madhekar/work/home-media-app/data/input-data/img'
#     ibtf = init()
#     for img in os.listdir(img_path):
#       nfaces, names = pred_names(ibtf, os.path.join(img_path, img))
#       print(f' file: {img} # of faces: {nfaces} identified faces: {names}')

# # kick-off face training generation
# if __name__ == "__main__":
#     exec()
    