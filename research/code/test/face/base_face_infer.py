import cv2 as cv
import joblib
import numpy as np
from mtcnn.mtcnn import MTCNN
from ast import literal_eval
import base_facenet as bfn

class infer_faces:
    def __init__(self, _faces_embeddings, _faces_label_enc, _faces_model_svc):
        self.detector = None 
        self.t_size = (160, 160)
        self.x = None
        self.y = None
        self._faces_embeddings = _faces_embeddings
        self._faces_label_enc = _faces_label_enc 
        self._faces_model_svc = _faces_model_svc

    def init(self):    
        self.detector = MTCNN()
        self.x, self.y =  np.load(self._faces_embeddings)
        self.faces_model_svc =  joblib.load(self._faces_model_svc)
        self.faces_label_enc =  joblib.load(self._faces_label_enc)
        self.facenet = bfn.base_facenet()

    def extract_faces(self, img):
        dict = {}
        tin =  cv.imread(img)
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
            if cnt == 1:  
               names.append('and a person')  
            else:    
               names.append(f'and {cnt} people')
              
        return names       

    def predict_names(self, img):

        nfaces = 0
        names = []
        try:
            if img:
                dict =  self.extract_faces(img)

                if dict and len(dict.keys()) > 0:
                    nfaces= len(dict.keys())
                    names = []
                    for key in dict:
                        test_im =  self.facenet.get_embeddings(dict[key])
                        test_im = [test_im]

                        ypred = self.faces_model_svc.predict(test_im)
                        names.append(''.join(literal_eval(str(self.faces_label_enc.inverse_transform(ypred)))))
                    names = self.replace_duplicates_and_missing(nfaces, names)
        except Exception as e:
            print(f'Excetion : {e} ')
        return ', '.join(names)        

