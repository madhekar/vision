from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageDraw
import os

class base_facenet:
  def __init__(self):  
    self.mtcnn = MTCNN(keep_all=True)
    self.resnet = InceptionResnetV1(pretrained="casia-webface").eval()
    print(self.resnet)

  def get_embeddings(self, face_img):
         aligned = self.mtcnn(face_img)
         embeddings = self.resnet(aligned).detach()
         return embeddings
       
if '__name__' == '__main__':
    img = Image.open(os.path.join("/home/madhekar/work/home-media-app/data/input-data/img", "Collage8.jpg") )
    bf = base_facenet()
    print(bf.get_embeddings(img))
