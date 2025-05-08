import os
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import matplotlib.pyplot as plt


class bface:
    def __init__(self, _dir):
        self.dir = _dir
        self.t_size = (160, 160)
        self.x = []
        self.y = []
        self.mtcnn = MTCNN(image_size=160, keep_all=True)
        self.resnet = InceptionResnetV1(pretrained="casia-webface").eval()
    '''
        # boxes, _ = mtcnn.detect(img)

        # if boxes is not None:
        #     for box in boxes:
        #         draw = ImageDraw.Draw(img)
        #         draw.rectangle(box.tolist(), outline='red', width=1)
        # img.show()        
        # if boxes is not None:
    '''
    def extract_face(self, fn):
        img = Image.open(fn)
        face_cropped = self.mtcnn.detect(img)
        x,y,w,h = r[0],r[1],r[2],r[3]
        print(x, y, w, h)
        x, y = abs(x), abs(y)
        face = img[y : y + h, x : x + w]
        face_arr = Image.resize(face, self.t_size)
        return face_arr

    def load_faces(self, dir):
        faces = []
        for im_file in os.listdir(dir):
            try:
                fp = os.path.join(dir, im_file)
                single_f = self.extract_face(fp)
                faces.append(single_f)
            except Exception as e:
                print(f"exception occreed {e}")
        return faces

    def load_names_and_faces(self):
        for sub_dir in os.scandir(self.dir):
            path = os.path.join(self.dir, sub_dir)
            print(sub_dir)
            faces = self.load_faces(path)
            print(len(faces))
            labels = [sub_dir.name for _ in range(len(faces))]
            self.x.extend(faces)
            self.y.extend(labels)
        return np.asarray(self.x), np.asarray(self.y)

    def plot_images(self):
        plt.figure(figsize=(20, 20))
        for num, img in enumerate(self.x):
            ncols = 4
            nrows = len(self.y) // ncols + 1
            plt.subplot(nrows, ncols, num + 1)
            plt.imshow(img)
            plt.axis("off")
        plt.show()
