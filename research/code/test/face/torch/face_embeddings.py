from facenet_pytorch import MTCNN, InceptionResnetV1

from PIL import Image, ImageDraw
import os

mtcnn = MTCNN(keep_all=True)

resnet = InceptionResnetV1(pretrained='casia-webface').eval()

img = Image.open(os.path.join('/home/madhekar/work/home-media-app/data/input-data/img', 'Collage8.jpg'))

# boxes, _ = mtcnn.detect(img)

# if boxes is not None:
#     for box in boxes:
#         draw = ImageDraw.Draw(img)
#         draw.rectangle(box.tolist(), outline='red', width=1)
# img.show()        
# if boxes is not None:
aligned = mtcnn(img)
embeddings = resnet(aligned).detach()

print(embeddings[0])