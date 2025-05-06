import os
import base_face_infer as bft
import multiprocessing as mp
import concurrent.futures 


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
    
   

def worker(img):
   names = bfs.pred_names_of_people(img)
   return names

"""
import concurrent.futures
import multiprocessing
import time

    cores = multiprocessing.cpu_count()

    with concurrent.futures.ProcessPoolExecutor(max_workers=cores) as executor:
        res = executor.map(task_1, nums)

        for r in res:
            print(r)


"""
# pool iniitializer
def pool_init(BFS):
    global bfs
    bfs = BFS

def exec():
    img_path = '/home/madhekar/work/home-media-app/data/input-data/img'
    cores =  mp.cpu_count()
    BFS = base_face_res()
    BFS.init() 
    pool_init(BFS)


    imgs = [os.path.join (img_path, ifile) for ifile in os.listdir(img_path)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=cores, initializer=pool_init, initargs=(BFS,)) as executor:
       res = executor.map(worker, imgs)
       for r in res:
           print(r)
    # for img in os.listdir(img_path):
    #   p = mp.Process(target=worker, args=(img,))
    #   processes.append(p)
    #   #names =  bfs.pred_names_of_people( os.path.join(img_path, img))
    #   p.start()
    #   p.join()
    #   names = p.name
    #   print(f' file: {img} identified faces: {names}')
   
if __name__=="__main__":
  exec()