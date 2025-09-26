import cv2 as cv
import joblib
import numpy as np
from mtcnn.mtcnn import MTCNN
from ast import literal_eval
from utils.face_util import base_facenet as bfn
import streamlit as st


class infer_faces:
    def __init__(self, _faces_embeddings, _faces_label_enc, _faces_model_svc):
        self.detector = None 
        self.t_size = (160, 160)
        self.x = None
        self.y = None
        self.faces_embeddings = _faces_embeddings
        self.faces_label_enc = _faces_label_enc 
        self.faces_model_svc = _faces_model_svc

    def init(self):    
        self.detector = MTCNN()
        self.x, self.y = np.load(self.faces_embeddings)
        self.faces_model_svc = joblib.load(self.faces_model_svc)
        self.faces_label_enc = joblib.load(self.faces_label_enc)
        self.facenet = bfn.base_facenet()

    def extract_faces(self, img):
        dict = {}
        try:
            tin = cv.imread(img)
            tin = cv.cvtColor(tin, cv.COLOR_BGR2RGB)

            res = self.detector.detect_faces(
                tin, 
                detector_backend="retinaface",  # "opencv", # Use a detector backend
                enforce_detection=False # Set to False to prevent errors if no faces are found
            )

            if res and len(res) > 0:
                dict = {}
                cnt = 1
                for d in res:
                    x, y, w, h = d["box"]
                    face = tin[y : y + h, x : x + w]
                    face_arr = cv.resize(face, (160, 160))
                    print('--->', str(x), str(y))
                    key = f"{str(x)}:{str(y)}"
                    dict[key] = face_arr
                    cnt += 1
        except Exception as e:
            print(f'extract_faces detected an exception: {e}')
            
        return dict

    def replace_duplicates_and_missing(self, nfaces, names, prefix='person-'):
        seen = set()
        cnt = 0
        for i in range(len(names)):
            if names[i] in seen:
                cnt +=1
                names[i] = prefix + str(cnt)
            else:
                seen.add(names[i])
        
        if nfaces > len(names):
            nmissing = nfaces - len(names)
            for j in range(nmissing):
                cnt +=1 
                names.append(prefix + str(cnt)) 
   
        if cnt > 0:
            names = [name for name in names if not name.startswith('person-')] 
            txt = 'and a person' if cnt == 1 else f'and {cnt} people'
            names.append(f'{txt}')   

        return names 

    def predict_names(self, img):
        nfaces = 0
        names = []
        try:
            if img:
                dict = self.extract_faces(img)
                #print('---->', dict)
                if dict and len(dict.keys()) > 0:
                    nfaces= len(dict.keys())
                    names = []
                    for key in dict:
                        test_im = self.facenet.get_embeddings(dict[key])
                        test_im = [test_im]

                        ypred = self.faces_model_svc.predict(test_im)
                        probs = self.faces_model_svc.predict_proba(test_im)
                        confidence  = np.max(probs)
                        if confidence < 0.6:
                            predicted_person = "unknown"
                        else:
                            predicted_person = literal_eval(str(self.faces_label_enc.inverse_transform(ypred)))
                        names.append(''.join(predicted_person))
                    names = self.replace_duplicates_and_missing(nfaces, names)
                    print('--->', names)
        except Exception as e:
            print(f'Excetion : {e} ')
        return nfaces, names        


