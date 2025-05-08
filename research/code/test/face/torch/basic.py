from facenet_pytorch import MTCNN
import cv2
import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm

mtcnn = MTCNN(
    #margin=40,
    #select_largest=False,
    post_process=False,
)

# # Load a single image and display
# v_cap = cv2.VideoCapture(
#     "/kaggle/input/deepfake-detection-challenge/train_sample_videos/agqphdxmwt.mp4"
# )
# success, frame = v_cap.read()

def basic_single_mtcnn(img_path, img_file):
    img = cv2.imread(os.path.join(img_path, img_file))
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)

    # Detect face
    face = mtcnn(frame)
    print(face.shape)

    plt.subplot(1,2,1)

    plt.suptitle("full_image")
    plt.imshow(frame)
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.suptitle("face_image")
    print(f'-->{face.shape}, {face.ndim}')
    plt.imshow(face.permute(1, 2, 0).int().numpy())
    plt.axis("off")

    # Visualize
    plt.show()

def basic_multiple_faces_mtcnn(img_path, img_file):


# Load a single image and display
    img = cv2.imread(os.path.join(img_path, img_file))
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)

    # Detect face
    faces = mtcnn(frame)
    print(f'-->-{faces[0].shape}')
    #plt.figure(figsize=(12, 8))
    plt.imshow(frame)
    plt.axis('off')
    plt.show()

    print(len(faces), faces[0].shape)
    # Visualize
    fig, axes = plt.subplots(1, len(faces))
    for face, ax in zip(faces, axes):
        #face = face.unsqueeze(1)
        print(f'{face.shape}, {face.ndim}')
        ax.imshow(face.permute(1, 2, 0).int().numpy())
        ax.axis('off')
    fig.show()    

basic_single_mtcnn("/home/madhekar/work/home-media-app/data/train-data/img/AnjaliBackup","imgIMG_3213.jpeg")    

basic_multiple_faces_mtcnn("/home/madhekar/work/home-media-app/data/train-data/img/AnjaliBackup","imgIMG_1529.jpeg")