import cv2 as cv
import joblib
import os
import numpy as np
from mtcnn.mtcnn import MTCNN

from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from utils.config_util import config
from utils.face_util import base_face as bf
from utils.face_util import base_facenet as bfn
import streamlit as st

def bface_train(faces_dir,  class_embeddings_folder, class_embeddings, label_encoder_path, label_encoder, faces_svc_path, faces_svc,sample_test_image):

    """
    load and embed
    """
    st.info('loading face database and extracting faces...')
    bface_inst = bf.bface(faces_dir)
    x, y = bface_inst.load_names_and_faces()

    # test faces load
    #bface_inst.plot_images()

    embedded_x = []
    b_fasenet = bfn.base_facenet()

    for img in x:
        embedded_x.append(b_fasenet.get_embeddings(img))
    embedded_x = np.asarray(embedded_x)

    # persist people faces embeddings and classes/ names
    np.savez_compressed(os.path.join(class_embeddings_folder, class_embeddings), embedded_x, y)

    """
    Label encoder
    """
    st.info('creating new labels for faces...')
    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)

    joblib.dump(encoder, filename=os.path.join(label_encoder_path, label_encoder))

    """
    train SVC
    """
    st.info('creating new model to classifiy faces...')
    detector = MTCNN()

    model = SVC(kernel="rbf", probability=True)
    model.fit(embedded_x, y)
    joblib.dump(model, filename=os.path.join(faces_svc_path, faces_svc))

    '''
    single face inference test
    '''
    t_im = cv.imread(sample_test_image)
    t_im = cv.cvtColor(t_im, cv.COLOR_BGR2RGB)
    x, y, w, h = detector.detect_faces(t_im)[0]["box"]

    t_im = t_im[y : y + h, x : x + w]
    t_im = cv.resize(t_im, (160, 160))
    test_im = b_fasenet.get_embeddings(t_im)

    model = joblib.load(filename=os.path.join(faces_svc_path, faces_svc))
    ypred = model.predict([test_im])

    st.info(f'{ypred}, {encoder.inverse_transform(ypred)}')

"""
static-metadata:
      faces_metadata_path: /home/madhekar/work/home-media-app/data/app-data/static-metadata/faces
image-data:
      input_image_path: /home/madhekar/work/home-media-app/data/input-data-1/img
model-path:
      faces_embbedings_path: /home/madhekar/work/home-media-app/models/faces_embbedings
      faces_embbedings: faces_embeddings_done_for_classes.npz
      faces_label_enc_path: /home/madhekar/work/home-media-app/models/faces_label_enc
      faces_label_enc: faces_label_enc.joblib
      faces_svc_path: /home/madhekar/work/home-media-app/models/faces_svc
      faces_svc: faces_model_svc.joblib
      faces_of_people_parquet_path: /home/madhekar/work/home-media-app/data/app-data/static-metadata/image_people
      faces_of_people_parquet: image_people.parquet
"""
def exec():

    (
        faces_dir,
        input_image_path,
        class_embeddings_folder,
        class_embeddings,
        label_encoder_path,
        label_encoder,
        faces_svc_path,
        faces_svc,
        faces_of_people_parquet_path, 
        faces_of_people_parquet,
    ) = config.faces_config_load()
    # faces_dir,  class_embeddings_folder, class_embeddings, label_encoder_path, label_encoder, faces_svc_path, faces_svc = (
    #     "/home/madhekar/work/home-media-app/data/app-data/static-metadata/faces",
    #     "/home/madhekar/work/home-media-app/models/faces_embbedings",
    #     "faces_embeddings_done_for_classes.npz",
    #     "/home/madhekar/work/home-media-app/models/faces_label_enc",
    #     "faces_label_enc.joblib",
    #     "/home/madhekar/work/home-media-app/models/faces_svc",
    #     "faces_model_svc.joblib"
    # )
    sample_test_image = "/home/madhekar/work/home-media-app/data/input-data/img/imgIMG_2439.jpeg"
    bface_train(faces_dir,  class_embeddings_folder, class_embeddings, label_encoder_path, label_encoder, faces_svc_path, faces_svc, sample_test_image)     

# kick-off face training generation
if __name__ == "__main__":
    exec()