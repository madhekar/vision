import os
import time
import pandas as pd
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
        self.faces_svc= None,

    def init(self):
        (
            self.faces_dir ,  
            self.class_embeddings_folder,
            self.class_embeddings,
            self.label_encoder_path,
            self.label_encoder,
            self.faces_svc_path,
            self.faces_svc,
        ) = config.faces_config_load()

        self.faces_embeddings, self.faces_label_enc, self.faces_model_svc = (
            os.path.join(self.class_embeddings_folder, self.class_embeddings),
            os.path.join(self.label_encoder_path, self.label_encoder),
            os.path.join(self.faces_svc_path, self.faces_svc),
        )
        self.faces_infer_obj = bftf.infer_faces(
            self.faces_embeddings, self.faces_label_enc, self.faces_model_svc
        )
        self.faces_infer_obj.init()

    def pred_names_of_people(self, img):
        _,names =  self.faces_infer_obj.predict_names(img)    
        return ', '.join(names)


# def pred_names(ibtf, img):
#     names = ibtf.predict_names(img)
#     return names

@st.cache_resource
def init():
    (
        faces_dir,
        input_image_path,
        class_embeddings_folder,
        class_embeddings,
        label_encoder_path,
        label_encoder,
        faces_svc_path,
        faces_svc,
        faces_of_people_parquet_path, 
        faces_of_people_parquet,
    ) = config.faces_config_load()

    faces_embeddings, faces_label_enc, faces_model_svc = (
        os.path.join(class_embeddings_folder, class_embeddings),
        os.path.join(label_encoder_path,label_encoder),
        os.path.join(faces_svc_path,faces_svc)
        )
    ibtf = bftf.infer_faces(faces_embeddings, faces_label_enc, faces_model_svc)    
    ibtf.init()
    print(f'-->init face test{label_encoder_path}{faces_svc_path}')
    return ibtf

def process_images_in_batch(ibtf, parquet_file, img_dir, batch_size=10):
    img_paths= [os.path.join(img_dir, img) for img in os.listdir(img_dir)]
    num_imgs = len(img_paths)

    for i in range(0, num_imgs, batch_size):
        batch_paths = img_paths[i: i+ batch_size]
        result = {(file_path, ibtf.pred_names_of_people(file_path)) for file_path in batch_paths }
        df = pd.DataFrame(result, columns=["image", "people"])
        df.to_parquet(parquet_file, mode="append", engine="fastparquet")
        time.sleep(2)
    return num_imgs, 'Done!'

def exec():
    ibtf =  init()
    parquet_file = 'people_in_image.parquet'
    img_path = '/home/madhekar/work/home-media-app/data/input-data/img'
    num, ret = process_images_in_batch(ibtf, parquet_file, img_path,batch_size=100)
    st.info(f'processed {num} images to predict people with status: {ret}')

    #r = {(os.path.join(img_path,img_file), ibtf.pred_names_of_people(os.path.join(img_path, img_file))) for img_file in os.listdir(img_path)[0:2]}
    # df = pd.DataFrame(r, columns=['image', 'people'])
    # print(df)
    # df.to_parquet('./image_people.parquet')

    # for img in os.listdir(img_path):
    #   nfaces, names = pred_names(ibtf, os.path.join(img_path, img))
    #   print(f' file: {img} # of faces: {nfaces} identified faces: {names}')

# kick-off face training generation
if __name__ == "__main__":
    exec()
    