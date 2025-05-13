import os
from deepface import DeepFace
from collections import Counter
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import cv2
import time
import pandas as pd
import streamlit as st
from utils.face_util import base_face_infer as bftf
from utils.config_util import config
from utils.util import model_util as mu
from utils.util import fast_parquet_util as fpu

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
    ibtf = base_face_res(faces_dir, class_embeddings_folder, class_embeddings, label_encoder_path, label_encoder, faces_svc_path, faces_svc)
    #ibtf = bftf.infer_faces(faces_embeddings, faces_label_enc, faces_model_svc)    
    ibtf.init()
    st.info(f'init face predict {label_encoder_path}:{faces_svc_path}')
    return ibtf, input_image_path, faces_of_people_parquet_path, faces_of_people_parquet

def group_attribute_into_ranges(data, ranges_dict):
    return{k: len([n for n in data if r[0] <= n <= r[1]]) for k,r in ranges_dict.items()}

def compute_aggregate_msg(in_arr):
    str_age, str_emo, str_gen, str_race = "","","",""
    age_ranges = {'infant':(0,2), 'toddler': (3,5), 'child':(6,9), 'adolescent':(10,24), 'young adult':(25,39), 'middle adult':(40,64), 'elderly':(65,120)}
    #print(in_arr)
    if in_arr:
        if len(in_arr) > 0:
            df = pd.DataFrame(in_arr, columns=['age','emotion','gender','race'])
            #print(df.head())

            #age range
            age_data = df['age'].values.tolist()
            age_classify = group_attribute_into_ranges(age_data, age_ranges)
            str_age = ', '.join([ f'{v} {k}' for k,v in age_classify.items() if v > 0])
            #print(f'-->{str_age}')

            #common emotion
            emotion_data = df["emotion"].values.tolist()
            emo_cnt = Counter(emotion_data)
            str_emo = ', '.join([f'{v} {k}' for k, v in emo_cnt.items()])
            #print(f'-->{str_emo}')

            #male count vs female count
            gender_data = df['gender'].values.tolist()
            gen_cnt = Counter(gender_data)
            str_gen = ", ".join([f"{v} {k}" for k, v in gen_cnt.items()])
            #print(f'-->{str_gen}')

            #race common race
            race_data = df['race'].values.tolist()
            race_cnt = Counter(race_data)
            str_race = ", ".join([f"{v} {k}" for k, v in race_cnt.items()])
            #print(f'-->{str_race}')
    return str_age + ' ' + str_emo + ' ' + str_gen + ' ' + str_race

def detect_human_attributs(img_path):
    people= []
    age, emotion, gender, race = None, None, None, None
    try:
        #print(img_path)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        preds = DeepFace.analyze(img, enforce_detection=True )
        #print(preds)
        if preds:
            num_faces = len(preds)
            if num_faces > 0:
                for nf in range(num_faces):
                    age = preds[nf]['age']
                    emotion = preds[nf]['dominant_emotion']
                    gender = preds[nf]["dominant_gender"]
                    race = preds[nf]["dominant_race"]
                    #print(f'{img_path}: {nf} of {num_faces} age: {age} - emotion: {emotion} - gender: {gender} - race: {race}')
                    people.append({'age':age, 'emotion': emotion, 'gender': gender, 'race': race})    
                    #print(people)
    except Exception as e:
        print(f'Error occured in emotion detection: {e}')
    return people  


def process_images_in_batch(ibtf, parquet_file, img_dir, batch_size=10):

    img_iterator = mu.getRecursive(img_dir, chunk_size=batch_size)
    st.info(f'processing images in {batch_size} batches: ')
    num_imgs = 0 
    for file_list in img_iterator:
        print(file_list)
        st.info('image processing batch in progress...')
        result = [[file_path, ibtf.pred_names_of_people(file_path), compute_aggregate_msg(detect_human_attributs(file_path))] for file_path in file_list ]
        df = pd.DataFrame(result, columns=["image", "people", "attrib"])
        #df.to_parquet(parquet_file, compression='snappy', append=True, index=None, engine="fastparquet")
        print(df.head())
        fpu.create_or_append_parquet(df, parquet_file)
        time.sleep(2)
        num_imgs += len(file_list)
    st.info(f'names of people from {num_imgs} images is complete!')    
    return num_imgs, 'Done!'

def exec(user_storage_name):
    st.info('predict names and attributes on people in images!')
    ibtf, img_path, faces_of_people_parquet_path, faces_of_people_parquet =  init()
    num, ret = process_images_in_batch(ibtf, os.path.join(faces_of_people_parquet_path, user_storage_name, faces_of_people_parquet), img_path, batch_size=10)
    st.info(f'processed {num} images to predict people with status: {ret}')

# kick-off face training generation
if __name__ == "__main__":
  exec('AnjaliBackup')
    