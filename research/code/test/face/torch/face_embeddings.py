from facenet_pytorch import MTCNN, InceptionResnetV1

from PIL import Image, ImageDraw
import os

def tensor_to_image(tensor):
      img = tensor.permute(1,2,0).detach().numpy()
      img = (img - img.min()) / (img.max() - img.min())
      img = (img * 255).astype('uint8')
      return img

def get_emb(pth, f):
    mtcnn = MTCNN(image_size=160,  keep_all=True)
    resnet = InceptionResnetV1(pretrained='casia-webface').eval()
    img = Image.open(os.path.join(pth, f))
    boxes, _ = mtcnn.detect(img)

    if boxes is not None:
        x = boxes[0].tolist()
        print(f"{x[0]} {x[1]} {x[2]} {x[3]}")
        for box in boxes:
            draw = ImageDraw.Draw(img)
            x = box.tolist()
            print(f"{x[0]}{x[1]}{x[2]}{x[3]}")
            draw.rectangle(box.tolist(), outline='red', width=1)
        img.show()  

img = Image.open(os.path.join("/Users/emadhekar/Downloads", "IMG_9067.PNG"))

e, f = get_emb(
    "/home/madhekar/work/home-media-app/data/train-data/img/AnjaliBackup",
    "imgIMG_3213.jpeg",
)    

print(e)

image = tensor_to_image(f)
dr = ImageDraw.Draw(image)
image.show()
