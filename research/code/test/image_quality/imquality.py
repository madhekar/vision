import imquality.brisque as brisque
from PIL import Image

img_path = "/home/madhekar/work/home-media-app/data/input-data-1/img/AnjaliBackup/0a6d1339-b672-50db-b5cb-984320fb90cc/vcm_s_kf_repr_886x587.jpg"
img = Image.open(img_path)
score = brisque.score(img)
print(f"BRISQUE Score: {score}")