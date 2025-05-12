import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import base_face_infer as bft
import pandas as pd
from deepface import DeepFace
import cv2

class base_face_res:
    def __init__(self):
        self.faces_embeddings = None,
        self.faces_label_enc = None,
        self.faces_model_svc = None
        self.faces_infer_obj = None

    def init(self):    
        self.faces_embeddings, self.faces_label_enc, self.faces_model_svc = (
            os.path.join("/home/madhekar/work/home-media-app/models/faces_embbedings","faces_embeddings_done_for_classes.npz"),
            os.path.join("/home/madhekar/work/home-media-app/models/faces_label_enc","faces_label_enc.joblib",),
            os.path.join("/home/madhekar/work/home-media-app/models/faces_svc","faces_model_svc.joblib")
        )

        self.faces_infer_obj = bft.infer_faces(self.faces_embeddings, self.faces_label_enc, self.faces_model_svc)
        self.faces_infer_obj.init()

    def pred_names_of_people(self, img):
        names =  self.faces_infer_obj.predict_names(img)
        return (names)
    
def detect_human_attributs(img_path):
    people = []
    age, emotion, gender, race = None, None, None, None
    try:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        preds = DeepFace.analyze(img, enforce_detection=True)

        if preds:
            num_faces = len(preds)
            if num_faces > 0:
                for nf in range(num_faces):
                    age = preds[nf]["age"]
                    emotion = preds[nf]["dominant_emotion"]
                    gender = preds[nf]["dominant_gender"]
                    race = preds[nf]["dominant_race"]
                    # print(f'{img_path}: {nf} of {num_faces} age: {age} - emotion: {emotion} - gender: {gender} - race: {race}')
                    people.append(
                        {"age": age, "emotion": emotion, "gender": gender, "race": race}
                    )
    except Exception as e:
        print(f"Error occured in emotion detection: {e}")
    return people    
def main():
    BFS = base_face_res()
    BFS.init() 
    fpath = '/home/madhekar/work/home-media-app/data/train-data/img/AnjaliBackup'
    r = {os.path.join(fpath, file) for file in os.listdir(fpath)}
    df = pd.DataFrame(r, columns=['image'])
    df['people'] = df.apply(lambda row: BFS.pred_names_of_people(row['image']), axis=1)
    df['age'], df['emotion'], df['gender'], df['race'] = df.apply(lambda row: detect_human_attributs(row['image']), axis=1)
    print(df)
    df.to_parquet('./image_people.parquet')

if __name__=="__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    tf.device('/cpu:0')
    print(tf.config.experimental.list_physical_devices())

    main()