import os
import base_face_infer as bft
import multiprocessing as mp

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
    
bfs = base_face_res()
bfs.init()    

def worker(img):
   names = bfs.pred_names_of_peope(img)
   return names

def exec():
    img_path = '/home/madhekar/work/home-media-app/data/input-data/img'
    processes = []
    for img in os.listdir(img_path):
      p = mp.Process(target=worker, args=(img,))
      processes.append(p)
      #names =  bfs.pred_names_of_people( os.path.join(img_path, img))
      p.start()
      p.join()
      names = p.name
      print(f' file: {img} identified faces: {names}')
   
if __name__=="__main__":
  exec()