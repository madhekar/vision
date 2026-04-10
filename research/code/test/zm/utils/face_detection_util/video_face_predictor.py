import os
#import cv2
from PIL import Image
#import pickle
#from ast import literal_eval
import joblib
import pandas as pd
import numpy as np
from insightface.app import FaceAnalysis
from deepface import DeepFace
#from utils.config_util import config
#import streamlit as st

def aggregate_partial_prompt(d):
    top_emotion = ""
    emo_rank = {'happy':1, 'surprise':2 ,'neutral':3, 'sad':4, 'fear':5, 'angry':6, 'disgust':7}
    df = pd.DataFrame(data=d)

    freq = df['emotion'].value_counts()
    df['freq'] = df['emotion'].map(freq)
    df['freq_rank'] = df['freq'].rank(method='dense', ascending=False).astype(int)
    df['emo_rank'] = df['emotion'].map(emo_rank)

    df_sorted = df.sort_values(by=['emo_rank', 'freq_rank'], ascending=[False, True])
    df_sorted['rank'] =   df_sorted['freq_rank'] + df_sorted['emo_rank'] 
    top_ranked = df_sorted['rank'].min()

    df_top_emo = df_sorted[df_sorted['rank'] == top_ranked]['emotion']

    print(f"sorted: {df_sorted.head(10)} top: {df_top_emo.head(10)}")
    if not df_top_emo.empty:
        top_emotion = df_top_emo.iloc[0]
    return top_emotion


def count_people(ld):
    cnt_man = 0
    cnt_woman = 0
    cnt_boy = 0
    cnt_girl = 0
    cnt_soul = 0

    agg_person = []
    for d in ld:
        if d["name"] == "unknown":
            if d["cnoun"] == "man":
                cnt_man += 1
            if d["cnoun"] == "woman":
                cnt_woman += 1
            if d["cnoun"] == "boy":
                cnt_boy += 1
            if d["cnoun"] == "girl":
                cnt_girl += 1
            if d['cnoun'] == 'soul':
                cnt_soul += 1    
        else:
            agg_person.append({"type": "known", "name": d["name"], "cnoun": d["cnoun"],"emotion": d["emotion"], "loc": d["loc"]})

    agg_person.append({"type": "unknown","cman": cnt_man, "cwoman": cnt_woman,"cboy": cnt_boy,"cgirl": cnt_girl, "csoul": cnt_soul})

    return agg_person


def create_partial_prompt(agg):
    txt = ""
    for d in agg:
        if d["type"] == "known":
            s = f' "{d["name"]}" ' #, a {d["emotion"]} {d["cnoun"]} '#, at {d["loc"]} '  #f'Face at coordinates {d["loc"]} is of "{d["name"]}", a "{d["cnoun"]}" expressing "{d["emotion"]}" emotion. '
            txt += s
        if d["type"] == "unknown":
            # if txt != "":
            #   txt += " and "
            if d["cman"] > 0:
                if d["cman"] > 1:
                    s = f"{d['cman']} men"
                else:
                    s = "one man"
                txt += s

            if d["cwoman"] > 0:
                if d["cwoman"] > 1:
                    s = f"{d['cwoman']} women"
                else:
                    s = "one  woman"
                txt += s

            if d["cboy"] > 0:
                if d["cboy"] > 1:
                    s = f" {d['cboy']} boys "
                else:
                    s = "one boy"
                txt += s

            if d["cgirl"] > 0:
                if d["cgirl"] > 1:
                    s = f" {d['cgirl']} girls "
                else:
                    s = "one girl"
                txt += s 

            if d['csoul'] > 0:
                if d['csoul'] > 1:
                    s = f"{d['csoul']} persons"
                else:
                    s = "one person"     
                txt += s 
            #txt += "in the image."
    return txt

""" 
Initialize InsightFace model
static-metadata:
      faces_metadata_path: /data/app-data/static-metadata/faces/training/images
model-path:
      faces_label_enc_path: /models/faces_label_enc
      faces_label_enc: faces_label_enc.joblib
      faces_svc_path: /models/faces_svc
      faces_svc: faces_model_svc.joblib

"""
#@st.cache_resource(ttl=36000, show_spinner=True)
def init_predictor_module():
    # (
    #     faces_dir,
    #     label_encoder_path,
    #     label_encoder,
    #     faces_svc_path,
    #     faces_svc,
    # ) = config.faces_config_load()
    label_encoder_path = "/mnt/zmdata/home-media-app/models/faces_label_enc"
    label_encoder = "faces_label_enc.joblib"
    faces_svc_path = "/mnt/zmdata/home-media-app/models/faces_svc"
    faces_svc = "faces_model_svc.joblib"
    
    label_encoder_store = os.path.join(label_encoder_path, label_encoder)
    svc_model_store = os.path.join(faces_svc_path, faces_svc)

    svm_classifier = joblib.load(svc_model_store)
    le = joblib.load(label_encoder_store)

    app = FaceAnalysis(name="buffalo_l",providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=-1, det_size=(256, 256)) # (640,640)

    return app, svm_classifier, le

'''
predict known and unknown faces
'''
def predict_img_faces(app, new_img_arr, svm_classifier, le):
    llm_partial_pmt = ""
    top_emo = ""
    #new_img = cv2.imread(new_image_path)
    # new_img =np.asarray(Image.open(new_image_path).convert('RGB'))
    people = []
    # Get face embedding
    faces = app.get(new_img_arr)
    if faces:
        print(f"Detected {len(faces)} face(s).")
        for i, face in enumerate(faces):
            person = {}

            if "gender" in face and "age" in face:
                gender = "Male" if face.gender == 1 else "Female"
                # detect age and gender for the face
                age = face.age
                person['age'] = age
                person['gender'] = gender
                if age > 21:
                    if gender == "Female":
                        person["cnoun"] = "soul" #"woman"
                    else:
                        person["cnoun"] = "soul" #"man"  
                else:
                    if gender == "Female":
                        person["cnoun"] = "soul" #"girl"
                    else:
                        person["cnoun"] = "soul" #"boy"    

            new_embedding = faces[i].embedding

            # Predict the identity using the trained SVM
            prediction = svm_classifier.predict([new_embedding])[0]
            class_probabilities = svm_classifier.predict_proba([new_embedding])[0]
            
            #print('--->',prediction, '::',le.inverse_transform([prediction]), ':', le.classes_)
            # Get the predicted class label
            predicted_person = le.inverse_transform([prediction])[0]
            #literal_eval(str(le.inverse_transform([prediction])[0]).strip())
            confidence = np.max(class_probabilities)
            if confidence < 0.5:
                predicted_person = "unknown"
            else:
                predicted_person = predicted_person

            # name for face in the image
            person['name'] = predicted_person

            x1, y1, x2, y2 = face.bbox.astype(int)
            cropped_face = new_img_arr[y1:y2, x1:x2]

            #face location in an image
            person['loc'] = (x1, y1)
            
            # "sad," "angry," "surprise," "fear," "happy," "disgust," and "neutral"
            em = DeepFace.analyze(cropped_face,actions=['emotion'],enforce_detection=False)
            if em:
                #person["emotion"] = em[0]["dominant_emotion"]
              emotion = em[0]["dominant_emotion"]
              if emotion == "angry" or emotion == "sad":
                  emotion = "neutral"
              person["emotion"] = emotion
            else:
              person["emotion"] = "neutral"    

            people.append(person)
        print(people)

        agg_cnt = count_people(people)
        
        if agg_cnt != "":
          llm_partial_pmt = create_partial_prompt(agg_cnt)
          top_emo  = aggregate_partial_prompt(person)
          #print((llm_partial_pmt, top_emo))
    else:
        pass
        print("No face detected in the image.")

        
    return (llm_partial_pmt, top_emo)


def execute():

    # Load a new image for recognition
    img = "/home/madhekar/work/home-media-app/data/input-data/img/madhekar/2596441a-e02f-588c-8df4-dc66a133fc99/af23fb36-9e89-4b81-9990-020b02fe1056.jpg"
    # "/home/madhekar/work/home-media-app/data/input-data/img/Samsung_USB/b6f657c7-7b7f-5415-82b7-e005846a6ef5/f6b572ec-a56f-4d66-b900-fa61b48ce005.jpg"
    # "/home/madhekar/work/home-media-app/data/input-data/img/Samsung_USB/b6f657c7-7b7f-5415-82b7-e005846a6ef5/7e9a4cc3-b380-40ff-a391-8bf596f8cd27.jpg"
    # "/home/madhekar/work/home-media-app/data/input-data/img/Samsung_USB/2a98fafb-a921-519f-8561-ed25ccd997de/5e144618-aea4-4365-95ad-375ae00a1133.jpg"

    # init train model
    app, svm_classifier, le = init_predictor_module()

    # train model using faces dataset
    print(predict_img_faces(app, img, svm_classifier, le))


if __name__ == "__main__":
    execute()


