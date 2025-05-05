import os
import base_face_infer as bft


class base_face_res:
    def __init__(self):
        self.faces_infer_obj = bft.infer_faces(
            os.path.join(
                "/home/madhekar/work/home-media-app/models/faces_embbedings",
                "faces_embeddings_done_for_classes.npz",
            ),
            os.path.join(
                "/home/madhekar/work/home-media-app/models/faces_label_enc",
                "faces_label_enc.joblib",
            ),
            os.path.join(
                "/home/madhekar/work/home-media-app/models/faces_svc",
                "faces_model_svc.joblib",
            ),
        )

    def pred_names_of_people(self, img):
        names = self.faces_infer_obj.predict_names(img)
        return names




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

def exec():
    img_path = '/home/madhekar/work/home-media-app/data/input-data/img'
    bfs = base_face_res()
    for img in os.listdir(img_path):
      nfaces, names = bfs.pred_names_of_people( os.path.join(img_path, img))
      print(f' file: {img} # of faces: {nfaces} identified faces: {names}')

exec()    