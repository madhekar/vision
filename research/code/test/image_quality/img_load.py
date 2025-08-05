import os
from PIL import Image
#from skimage import io, img_as_float
import matplotlib.pyplot as plt
import imquality.brisque as b

#import cv2.quality as q
import io

def load_image(image_path):
    try:
        # with open(image_path, 'rb') as img_bin:
        #     buff = io.BytesIO(img_bin.read())
        #     img = Image.open(buff).convert('RGB')
        #     return img
        i = Image.open(image_path).convert('RGB')
        return i
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def is_valid_brisque_score(image, brisque_model_path, model_live_file, model_range_file, threshold=50.0):
        if image is not None:            
            
            # _image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # gray = cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY)
            print(image.size)
            h, w = image.size
            if w < 512 or h < 512:
                return False

            brisque_score = b.score(image)
            # q.QualityBRISQUE_compute(
            #     image,
            #     os.path.join(brisque_model_path, model_live_file),
            #     os.path.join(brisque_model_path, model_range_file),
            # )
            print(f'brisque score: {brisque_score}')
            return brisque_score < threshold
        else:
            print("quality", "e| unable to load - NULL image")

        return False


def load_img(image_path):
    img = io.imread(image_path, as_gray=True, multichannel=False) #img_as_float(io.imread(image_path, as_gray=True))
    return img

# Example usage:
#image = load_image("/Users/emadhekar/Downloads/1a5e9da6-462d-4f2f-a289-5c45e0db1176.JPG")


brisque_model_config_path= '/home/madhekar/work/home-media-app/models/brisque'
brisque_model_live_file= 'brisque_model_live.yml'
brisque_range_live_file= 'brisque_range_live.yml'


image = load_image("/home/madhekar/work/home-media-app/data/input-data/img/AnjaliBackup/0a6d1339-b672-50db-b5cb-984320fb90cc/vcm_s_kf_repr_886x587.jpg")
if image:
    # Process the image
    bv = is_valid_brisque_score(image, brisque_model_config_path, brisque_model_live_file, brisque_range_live_file)
    plt.imshow(image)
    plt.show()