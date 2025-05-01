import numpy as np
from keras_facenet import FaceNet


class base_facenet:
    def __init__(self):
        self.embedder = FaceNet()

    def get_embeddings(self, face_img):
        face_img = face_img.astype("float32")
        face_img = np.expand_dims(face_img, axis=0)
        yhat = self.embedder.embeddings(face_img)
        return yhat[0]
