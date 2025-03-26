from PIL import Image, ImageOps
import imquality.brisque as b
from skimage import io, img_as_float
import os
from skimage.transform import rescale
from skimage.transform import resize
    


sample_path = "/home/madhekar/work/home-media-app/data/train-data/img/AnjaliBackup"

for rt, _, files in os.walk(sample_path, topdown=True):
    for file in files:

        # image = img_as_float(io.imread(os.path.join(rt, file), as_gray=True))
        
        

        image = Image.open(os.path.join(rt, file)).convert('L')
        
        #image_rescaled = image.resize((224, 224))
        #image_rescaled = ImageOps.grayscale(image)
        #image_rescaled = resize(image, (224, 224))

        score_brisque = b.score(image)

        print(
            f"file name: {file} nima score: {score_brisque:.4f}"
        )  # brisque score: {score_brisque.item():.4f}")
