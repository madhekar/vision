import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import base_face_infer as bft
import multiprocessing as mp
import concurrent.futures 
import aiomultiprocess as aiomp
from aiomultiprocess import Pool
import pandas as pd
import asyncio
import time



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
    

# def worker(img):
#     print('-->')
#     BFS = base_face_res()
#     BFS.init() 
#     return BFS.pred_names_of_people(img)


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


# async def main():
#     # BFS = base_face_res()
#     # BFS.init()

#     async with Pool() as pool:
#         tasks = [
#             pool.apply(worker, [os.path.join('/home/madhekar/work/home-media-app/data/train-data/img/AnjaliBackup', 'IMG_0697.JPG')])
#         ]
#         results = await asyncio.gather(*tasks)
#         print(results)

# def main():
#     with Pool(processes=1) as pool:
#         r = pool.map(worker, [os.path.join('/home/madhekar/work/home-media-app/data/train-data/img/AnjaliBackup', 'IMG_0697.JPG')])
#         print(r)

def main():
    BFS = base_face_res()
    BFS.init() 
    fpath = '/home/madhekar/work/home-media-app/data/train-data/img/AnjaliBackup'
    r = {os.path.join(fpath, file) for file in os.listdir(fpath)}
    df = pd.DataFrame(r, columns=['image'])
    df['people'] = df.apply(lambda row: BFS.pred_names_of_people(row['image']), axis=1)
    print(df)
    df.to_parquet('./image_people.parquet')

    #print( BFS.pred_names_of_people(os.path.join('/home/madhekar/work/home-media-app/data/train-data/img/AnjaliBackup', 'IMG_0697.JPG')))

# def exec():
#     img_path = '/home/madhekar/work/home-media-app/data/train-data/img/AnjaliBackup'
#     # cores =  mp.cpu_count()
#     BFS = base_face_res()
#     BFS.init() 
#     # pool_init(BFS)
#     # futures = []

#     imgs = [os.path.join (img_path, ifile) for ifile in os.listdir(img_path)]
#     print(imgs[0:10])
#     # # with concurrent.futures.ThreadPoolExecutor(max_workers=cores, initializer=pool_init, initargs=(bfs,)) as executor:
#     # #    res = executor.map(worker, imgs[0:100])
#     # #    for r in res:
#     # #       print(r.result())

#     with Pool(processes=4, maxtasksperchild=1) as pool:
#          pool.map(worker, imgs[0:10])


#     for img in os.listdir(img_path):
#       time.sleep(1)
#       #p = mp.Process(target=worker, args=(img,))
#       #processes.append(p)
#       names =  BFS.pred_names_of_people( os.path.join(img_path, img))
#     #   p.start()
#     #   p.join()
#     #   names = p.name
#       print(f' file: {img} identified faces: {names}')
   
if __name__=="__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    tf.device('/cpu:0')
    #tf.config.set_visible_devices([], 'CPU')

    print(tf.config.experimental.list_physical_devices())
    # exec()
    #asyncio.run(main())
    main()