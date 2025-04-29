from pycocotools.coco import COCO
import os
from PIL import Image
from skimage.io import imsave
import numpy as np
from matplotlib import pyplot as plt


coco = COCO("/home/madhekar/work/vision/research/code/test/annotations/annotations.json")
img_dir = "/home/madhekar/work/vision/research/code/test/annotations/images"

def show_all_images():
    imgs =  coco.getImgIds()
    num = len(imgs)
    print(f'num of images: {num}')

    for ids in imgs:
        imgs = coco.loadImgs(ids=ids)
        print(f'->>>{imgs}')


def show_all_categories():
    cat_ids = coco.getCatIds()
    num = len(cat_ids)
    print(f'num of categoes: {num}')

    for ids in cat_ids:
        cats = coco.loadCats(ids=ids)
        print(cats)

def gen_coco_mask():
    #get all image ids
    imgs = coco.getImgIds()
    cat_ids = coco.getCatIds()
    for id in imgs:
        img = coco.imgs[id]
        print(img)
        ann_ids = coco.getAnnIds(img["id"], catIds=cat_ids, iscrowd=None)
        print(ann_ids)
        anns = coco.loadAnns(ann_ids)
        mask = coco.annToMask(anns[0])
        for i in range(len(anns)):
           mask += coco.annToMask(anns[i])
        if id == 1:
            plt.imsave('1.png', mask)

def show_coco_mask_for_image(image_id =74):

    img = coco.imgs[image_id]
    print(f'--->{img}')

    # image = np.array(Image.open(os.path.join(img_dir, img["file_name"])))
    # plt.subplot(1,3,1)
    # plt.imshow(image, interpolation="nearest")
    # plt.title('original image')
    #plt.show()
 
    # plt.subplot(1,3,2)
    # plt.imshow(image)

    cat_ids = coco.getCatIds()
    anns_ids = coco.getAnnIds(imgIds=img["id"], catIds=cat_ids, iscrowd=None)
    anns = coco.loadAnns(anns_ids)

    print(f'categories: {[ann["category_id"] for ann in anns]}')

    #coco.showAnns(anns)
    #plt.title("coco image annotation")
    #plt.show()

    #plt.subplot(1,3,3)
    mask = coco.annToMask(anns[0])
    for i in range(len(anns)):
        mask += coco.annToMask(anns[i])
    # plt.imshow(mask)
    # plt.title("image mask")
    # plt.show()
    plt.imsave('1.png', mask)


    # cat_ids = coco.getCatIds()
    # anns_ids = coco.getAnnIds(imgIds=img["id"], catIds=cat_ids, iscrowd=None)
    # anns = coco.loadAnns(anns_ids)
    # anns_img = np.zeros((img["height"], img["width"]))
    # for ann in anns:
    #     anns_img = np.maximum(anns_img, coco.annToMask(ann) * ann["category_id"])
    #     plt.imshow(anns_img)
    #     plt.show()

#show_all_categories()

#show_coco_mask_for_image(image_id=34)

#show_all_images()

gen_coco_mask()