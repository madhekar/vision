import os
import base_face_infer as bft


def predict_names(ibtf, img):
    names = ibtf.predict_names(img)
    return names

def init():
    faces_embeddings, faces_label_enc, faces_model_svc = (
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
        )
    )
    ibtf = bft.infer_faces(faces_embeddings, faces_label_enc, faces_model_svc)
    return ibtf

def exec():
    img_path = '/home/madhekar/work/home-media-app/data/input-data/img'
    ibtf = init()
    for img in os.listdir(img_path):
      print(img)
      nfaces, names = predict_names(ibtf, os.path.join(img_path, img))
      print(f'total faces detected: {nfaces} identified faces: {names}')

exec()    