from facenet_pytorch import MTCNN, InceptionResnetV1

from PIL import Image, ImageDraw
import os

def get_emb(pth, f):
    mtcnn = MTCNN(image_size=160,  keep_all=True)
    resnet = InceptionResnetV1(pretrained='casia-webface').eval()
    img = Image.open(os.path.join(pth, f))

    # boxes, _ = mtcnn.detect(img)

    # if boxes is not None:
    #     for box in boxes:
    #         draw = ImageDraw.Draw(img)
    #         draw.rectangle(box.tolist(), outline='red', width=1)
    # img.show()        
    # if boxes is not None:

    faces, probs = mtcnn(img, return_prob=True)
    if faces is None or len(faces) ==0:
        return None, None
    
    embeddings = resnet(faces[0].unsqueeze(0)).detach()
    print(embeddings, faces[0])

get_emb('/home/madhekar/work/home-media-app/data/input-data/img', 'Collage8.jpg')    