from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageDraw
import os

class base_facenet_pt:
  def __init__(self):  
      self.mtcnn = MTCNN(image_size=160, keep_all=True)
      self.resnet = InceptionResnetV1(pretrained="casia-webface").eval()
      print(self.resnet)

  def get_embeddings(self, face_img):
        faces, probs = self.mtcnn(face_img, return_prob=True)
        if faces is None or len(faces) ==0:
           return None, None
        
        embeddings = self.resnet(faces[0].unsqueeze(0)).detach()
        return embeddings
  
  def tensor_to_image(tensor):
      img = tensor.permute(1,2,0).detach().numpy()
      img = (img - img.min()) / (img.max() - img.min())
      img = (img * 255).astype('uint8')
      return img
       
if __name__ == '__main__':
    img = Image.open(os.path.join("/home/madhekar/work/home-media-app/data/input-data/img", "Collage8.jpg") )
    bf = base_facenet_pt()
    print(bf.get_embeddings(img))
