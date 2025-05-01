import os
import base_face_infer as bft


def predict_names(ibtf, img):
    names = ibtf.predict_names(img)
    return names

"""
        "/home/madhekar/work/home-media-app/data/app-data/static-metadata/faces",
        "/home/madhekar/work/home-media-app/models/faces_embbedings",
        "faces_embeddings_done_for_classes.npz",
        "/home/madhekar/work/home-media-app/models/faces_label_enc",
        "faces_label_enc.joblib",
        "/home/madhekar/work/home-media-app/models/faces_svc",
        "faces_model_svc.joblib"
"""
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
    ibtf = init()
    for img in os.listdir("/home/madhekar/work/home-media-app/data/input-data/img"):
      names = predict_names(ibtf)
      print(names)

exec()    