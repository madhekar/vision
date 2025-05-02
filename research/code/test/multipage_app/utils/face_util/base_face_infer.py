import cv2 as cv
import joblib
import numpy as np
from mtcnn.mtcnn import MTCNN
from ast import literal_eval
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

    def replace_duplicates_and_missing(self, nfaces, names, prefix='person-'):
        seen = set()
        cnt = 1
        for i in range(len(names)):
            if names[i] in seen:
                names[i] = prefix + str(cnt)
                cnt +=1
            else:
                seen.add(names[i])
        
        if nfaces > len(names):
            nmissing = nfaces - len(names)
            for j in range(nmissing):
                names.append(prefix + str(cnt)) 
                cnt +=1
        rnames = [name for name in names if not name.startswith("person-")]
        txt = "person" if cnt == 1 else "people"
        rnames.append(f"{cnt} {txt}")
        return rnames

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
                        names.append(''.join(literal_eval(str(self.faces_label_enc.inverse_transform(ypred)))))
                    names = self.replace_duplicates_and_missing(nfaces, names)
        except Exception as e:
            print(f'Excetion : {e} ')
        return nfaces, names        

