import os
import cv2
import pickle
from ast import literal_eval
import joblib
import pandas as pd
import numpy as np
from insightface.app import FaceAnalysis
from deepface import DeepFace
from utils.config_util import config
import streamlit as st

def count_people(ld):
    cnt_man = 0
    cnt_woman = 0
    cnt_boy = 0
    cnt_girl = 0

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
        else:
            agg_person.append({"type": "known", "name": d["name"], "cnoun": d["cnoun"],"emotion": d["emotion"], "loc": d["loc"]})

    agg_person.append({"type": "unknown","cman": cnt_man, "cwoman": cnt_woman,"cboy": cnt_boy,"cgirl": cnt_girl})

    return agg_person


def create_partial_prompt(agg):
    txt = ""
    for d in agg:
        if d["type"] == "known":
            s = f' "{d["name"]}", a {d["emotion"]} {d["cnoun"]}, at {d["loc"]} '  #f'Face at coordinates {d["loc"]} is of "{d["name"]}", a "{d["cnoun"]}" expressing "{d["emotion"]}" emotion. '
            txt += s
        if d["type"] == "unknown":
            txt += " and "
            if d["cman"] > 0:
                if d["cman"] > 1:
                    s = f" {d['cman']} men "
                else:
                    s = " one man "
                txt += s

            if d["cwoman"] > 0:
                if d["cwoman"] > 1:
                    s = f" {d['cwoman']} women "
                else:
                    s = " one  woman  "
                txt += s

            if d["cboy"] > 0:
                if d["cboy"] > 1:
                    s = f" {d['cboy']} boys "
                else:
                    s = " one boy "
                txt += s

            if d["cgirl"] > 0:
                if d["cgirl"] > 1:
                    s = f" {d['cgirl']} girls "
                else:
                    s = " one girl "
                txt += s 
            txt += "in the image."
    return txt

""" 
Initialize InsightFace model
"""
@st.cache_resource(ttl=36000, show_spinner=True)
def init_predictor_module():
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

    label_encoder_store = os.path.join(label_encoder_path, label_encoder)
    svc_model_store = os.path.join(faces_svc_path, faces_svc)

    svm_classifier = joblib.load(svc_model_store)
    le = joblib.load(label_encoder_store)

    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=-1, det_size=(640, 640))

    return app, svm_classifier, le

'''
predict known and unknown faces
'''
def predict_img_faces(app, new_image_path, svm_classifier, le):
    llm_partial_pmt = ""

    new_img = cv2.imread(new_image_path)
    people = []
    # Get face embedding
    faces = app.get(new_img)
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
                        person["cnoun"] = "woman"
                    else:
                        person["cnoun"] = "man"  
                else:
                    if gender == "Female":
                        person["cnoun"] = "girl"
                    else:
                        person["cnoun"] = "boy"    

            new_embedding = faces[i].embedding

            # Predict the identity using the trained SVM
            prediction = svm_classifier.predict([new_embedding])[0]
            class_probabilities = svm_classifier.predict_proba([new_embedding])[0]
            
            #print('--->',prediction, '::',le.inverse_transform([prediction]), ':', le.classes_)
            # Get the predicted class label
            predicted_person = le.inverse_transform([prediction])[0]
            #literal_eval(str(le.inverse_transform([prediction])[0]).strip())
            confidence = np.max(class_probabilities)
            if confidence < 0.6:
                predicted_person = "unknown"
            else:
                predicted_person = predicted_person

            # name for face in the image
            person['name'] = predicted_person

            x1, y1, x2, y2 = face.bbox.astype(int)
            cropped_face = new_img[y1:y2, x1:x2]

            #face location in an image
            person['loc'] = (x1, y1)

            em = DeepFace.analyze(cropped_face,actions=['emotion'],enforce_detection=False)
            # face emotion 
            person["emotion"] = em[0]["dominant_emotion"]

            people.append(person)
        print(people)

        agg_cnt = count_people(people)
        
        if agg_cnt != "":
          llm_partial_pmt = create_partial_prompt(agg_cnt)
          print(llm_partial_pmt)
    else:
        print("No face detected in the image.")
    return llm_partial_pmt


def execute():

    # Load a new image for recognition
    img = "/home/madhekar/work/home-media-app/data/input-data/img/madhekar/2596441a-e02f-588c-8df4-dc66a133fc99/af23fb36-9e89-4b81-9990-020b02fe1056.jpg"
    # "/home/madhekar/work/home-media-app/data/input-data/img/Samsung_USB/b6f657c7-7b7f-5415-82b7-e005846a6ef5/f6b572ec-a56f-4d66-b900-fa61b48ce005.jpg"
    # "/home/madhekar/work/home-media-app/data/input-data/img/Samsung_USB/b6f657c7-7b7f-5415-82b7-e005846a6ef5/7e9a4cc3-b380-40ff-a391-8bf596f8cd27.jpg"
    # "/home/madhekar/work/home-media-app/data/input-data/img/Samsung_USB/2a98fafb-a921-519f-8561-ed25ccd997de/5e144618-aea4-4365-95ad-375ae00a1133.jpg"

    # init train model
    app, svm_classifier, le = init_predictor_module()

    # train model using faces dataset
    predict_img_faces(app, img, svm_classifier, le)


if __name__ == "__main__":
    execute()


