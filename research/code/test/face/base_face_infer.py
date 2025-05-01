import cv2 as cv
import joblib
import numpy as np
from mtcnn.mtcnn import MTCNN

from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import base_facenet as bfn

class infer_faces:
    def __init__(self, _mpath, _ppath, _lpath):
        self.detector = MTCNN()
        self.t_size = (160, 160)
        self.x, self.y = np.load(_ppath)
        self.model = joblib.load(_mpath)
        self.label_encoder = joblib(_lpath)
        self.facenet = bfn.base_facenet()

    def extract_faces(self, img):
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
        if img:
          dict = self.extract_faces(img)

          if dict and dict.keys() > 0:
            names = []
            for e in dict.items():
                test_im = self.facenet.get_embeddings(e.value())
                test_im = [test_im]

                ypred = self.model.predict(test_im)
                names.append(self.label_encoder.inverse_transform(ypred))
        return names        

