import os
import base_face_infer as bft
import asyncio

class base_face_res:
    def __init__(self):
        self.faces_embeddings = None,
        self.faces_label_enc = None,
        self.faces_model_svc = None

    def init(self):    
        self.faces_embeddings, self.faces_label_enc, self.faces_model_svc = (
            os.path.join("/home/madhekar/work/home-media-app/models/faces_embbedings","faces_embeddings_done_for_classes.npz"),
            os.path.join("/home/madhekar/work/home-media-app/models/faces_label_enc","faces_label_enc.joblib",),
            os.path.join("/home/madhekar/work/home-media-app/models/faces_svc","faces_model_svc.joblib")
        )
        
        self.faces_infer_obj = bft.infer_faces(
            self.faces_embeddings, self.faces_label_enc, self.faces_model_svc
        )

    
    async def pred_names_of_people(self, img):
        _, names = self.faces_infer_obj.predict_names(img)
        return ' '.join(names)




# def predict_names(ibtf, img):
#     names = ibtf.predict_names(img)
#     return names

# def init():
#     faces_embeddings, faces_label_enc, faces_model_svc = (
#         os.path.join(
#             "/home/madhekar/work/home-media-app/models/faces_embbedings",
#             "faces_embeddings_done_for_classes.npz",
#         ),
#         os.path.join(
#             "/home/madhekar/work/home-media-app/models/faces_label_enc",
#             "faces_label_enc.joblib",
#         ),
#         os.path.join(
#             "/home/madhekar/work/home-media-app/models/faces_svc",
#             "faces_model_svc.joblib",
#         )
#     )
#     ibtf = bft.infer_faces(faces_embeddings, faces_label_enc, faces_model_svc)
#     return ibtf

async def exec():
    img_path = '/home/madhekar/work/home-media-app/data/input-data/img'
    bfs = base_face_res()
    bfs.init()
    for img in os.listdir(img_path):
      names = await bfs.pred_names_of_people( os.path.join(img_path, img))
      print(f' file: {img} identified faces: {names}')
   
if __name__=="__main__":
   asyncio.run(exec())