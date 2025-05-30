
import os

from collections import Counter
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import pandas as pd
import cv2
from deepface import DeepFace
import streamlit as st
from utils.face_util import base_face_infer as bftf
from utils.config_util import config
from utils.util import model_util as mu
from utils.util import fast_parquet_util as fpu

class base_face_res:

    def __init__(self, _faces_dir,_class_embeddings_folder,_class_embeddings, _label_encoder_path, _label_encoder, _faces_svc_path, _faces_svc):
        self.faces_dir = _faces_dir
        self.class_embeddings_folder= _class_embeddings_folder
        self.class_embeddings= _class_embeddings
        self.label_encoder_path= _label_encoder_path
        self.label_encoder= _label_encoder
        self.faces_svc_path= _faces_svc_path
        self.faces_svc= _faces_svc

    def initialize(self):
        print(self.class_embeddings_folder, self.class_embeddings)
        self.faces_infer_obj = bftf.infer_faces(
            os.path.join(self.class_embeddings_folder, self.class_embeddings),
            os.path.join(self.label_encoder_path, self.label_encoder),
            os.path.join(self.faces_svc_path, self.faces_svc)
        )
        self.faces_infer_obj.init()

    def pred_names_of_people(self, img):
        _,names =  self.faces_infer_obj.predict_names(img)    
        return ', '.join(names)

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
        faces_of_people_parquet
    ) = config.faces_config_load()

    print(class_embeddings, class_embeddings_folder)
    ibtf = base_face_res(faces_dir, class_embeddings_folder, class_embeddings, label_encoder_path, label_encoder, faces_svc_path, faces_svc)  
    print('-->hrer')
    ibtf.initialize()
    st.info(f'init face predict {label_encoder_path}:{faces_svc_path}')
    return ibtf, input_image_path, faces_of_people_parquet_path, faces_of_people_parquet

def group_attribute_into_ranges(data, ranges_dict):
    return{k: len([n for n in data if r[0] <= n <= r[1]]) for k,r in ranges_dict.items()}

def compute_aggregate_msg(in_arr):
    str_age, str_emo, str_gen, str_race = "","","",""
    age_ranges = {'infant':(0,2), 'toddler': (3,5), 'child':(6,9), 'adolescent':(10,24), 'young adult':(25,39), 'middle adult':(40,64), 'elderly':(65,120)}
    print(in_arr)
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
        print(f'---->{img_path}')
        # img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        preds = DeepFace.analyze(
            img_path,
            enforce_detection=False
        )
        print(preds)
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
                    print(f'---->{people}')
    except Exception as e:
        print(f'Error occurred in emotion detection: {e}')
    return people  

"""
deepface                                 0.0.93
tensorboard                              2.19.0
tensorboard-data-server                  0.7.2
tensorflow                               2.16.1
tensorflow_cpu                           2.19.0
"""
def process_images_in_batch(ibtf, parquet_file, img_dir, batch_size=1):

    # BFS = base_face_res()
    # BFS.init() 
    fpath = '/home/madhekar/work/home-media-app/data/train-data/img/AnjaliBackup'
    r = {os.path.join(fpath, file) for file in os.listdir(fpath)[0:1]}
    df = pd.DataFrame(r, columns=['image'])
    print(df)
    df['people'] = df.apply(lambda row: ibtf.pred_names_of_people(row['image']), axis=1)
    df['attrib'] =  df.apply(lambda row: compute_aggregate_msg(detect_human_attributs(row['image'])), axis=1)
    print(df)
    #df.to_parquet('./image_people.parquet')
    return df.size, 'Done!'


    # file_list = os.listdir('/home/madhekar/work/home-media-app/data/train-data/img/AnjaliBackup') #mu.getRecursive(img_dir, chunk_size=batch_size)
    # st.info(f'processing images in {batch_size} batches: ')
    # num_imgs = 0 
    # results = []
    # st.info('image processing batch in progress...')

    # for file_path in file_list[0:10]:
    #     names = ibtf.pred_names_of_people(os.path.join('/home/madhekar/work/home-media-app/data/train-data/img/AnjaliBackup' ,file_path))
    #     gc.collect()
    #     tf.keras.backend.clear_session()
    #     attribs = compute_aggregate_msg(detect_human_attributs(os.path.join('/home/madhekar/work/home-media-app/data/train-data/img/AnjaliBackup' ,file_path)))
    #     gc.collect()
    #     tf.keras.backend.clear_session()

    #     #result = [[file_path, ibtf.pred_names_of_people(file_path), compute_aggregate_msg(detect_human_attributs(file_path))] for file_path in file_list ]
    #     #result = {'image': file_path, 'names': names, 'attribs': attribs}
    #     print(names, attribs)
    #     num_imgs += 1
    #     results.append({"image": file_path, "names": names, "attribs": attribs})

    # df = pd.DataFrame(results)
    # #df.to_parquet(parquet_file, compression='snappy', append=True, index=None, engine="fastparquet")
    # print(df.head())
    # #fpu.create_or_append_parquet(df, parquet_file)
        
    # gc.collect()
    # st.info(f'names of people from {num_imgs} images is complete!')    
    # return num_imgs, 'Done!'

def exec(user_storage_name):
    st.info('predict names and attributes on people in images!')
    ibtf, img_path, faces_of_people_parquet_path, faces_of_people_parquet =  init()
    num, ret = process_images_in_batch(ibtf, os.path.join(faces_of_people_parquet_path, user_storage_name, faces_of_people_parquet), img_path, batch_size=1)
    st.info(f'processed {num} images to predict people with status: {ret}')

# kick-off face training generation
if __name__ == "__main__":
  exec('AnjaliBackup')
    