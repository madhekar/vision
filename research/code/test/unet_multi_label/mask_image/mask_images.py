from pycocotools.coco import COCO
import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt


coco = COCO("/home/madhekar/work/vision/research/code/test/annotations/annotations.json")
img_dir = "/home/madhekar/work/vision/research/code/test/annotations/images"
image_id = 74


img = coco.imgs[image_id]
print(img)

image = np.array(Image.open(os.path.join(img_dir, img["file_name"])))
plt.imshow(image, interpolation="nearest")
plt.show()

plt.imshow(image)
cat_ids = coco.getCatIds()
anns_ids = coco.getAnnIds(imgIds=img["id"], catIds=cat_ids, iscrowd=None)
anns = coco.loadAnns(anns_ids)
coco.showAnns(anns)
plt.show()

mask = coco.annToMask(anns[0])
for i in range(len(anns)):
    mask += coco.annToMask(anns[i])

plt.imshow(mask)
plt.show()
