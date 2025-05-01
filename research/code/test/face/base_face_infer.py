import cv2 as cv
import joblib
import numpy as np
from mtcnn.mtcnn import MTCNN
import time
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import base_facenet as bfn

class infer_faces:
    def __init__(self, _faces_embeddings, _faces_label_enc, _faces_model_svc):
        self.detector = MTCNN()
        self.t_size = (160, 160)
        self.x, self.y = np.load(_faces_embeddings)
        self.faces_model_svc = joblib.load(_faces_model_svc)
        self.faces_label_enc = joblib.load(_faces_label_enc)
        self.facenet = bfn.base_facenet()

    def extract_faces(self, img):
        dict = {}
        tin = cv.imread(img)
        tin = cv.cvtColor(tin, cv.COLOR_BGR2RGB)
        res = self.detector.detect_faces(tin)

        if res and len(res) > 0:
            dict = {}
            cnt = 1
            for d in res:
                x, y, w, h = d["box"]
                face = tin[y : y + h, x : x + w]
                face_arr = cv.resize(face, (160, 160))
                key = f"face_{cnt}"
                dict[key] = face_arr
                cnt += 1
        return dict

    def predict_names(self, img):
        nfaces = 0
        names = []
        try:
            if img:
                dict = self.extract_faces(img)

                if dict and len(dict.keys()) > 0:
                    nfaces= len(dict.keys())
                    names = []
                    for key in dict:
                        test_im = self.facenet.get_embeddings(dict[key])
                        test_im = [test_im]

                        ypred = self.faces_model_svc.predict(test_im)
                        names.append(self.faces_label_enc.inverse_transform(ypred))
        except Exception as e:
            print(f'Excetion : {e} ')
        return nfaces, names        

