import json
import numpy as np
import skimage
from skimage.io import imsave
import tifffile
from PIL import Image
import os
import shutil
from pycocotools.coco import COCO
from matplotlib import pyplot as plt




def create_mask(image_info, annotations, output_folder):
    # Create an empty mask as a numpy array
    mask_np = np.zeros((image_info["height"], image_info["width"], 3), dtype=np.uint8)

    # Counter for the object number
    object_number = 1

    for ann in annotations:
        if ann["image_id"] == image_info["id"]:
            # Extract segmentation polygon
            cat = ann['category_id']
            for seg in ann["segmentation"]:
                print(f'{cat} -- {seg}')
                # Convert polygons to a binary mask and add it to the main mask
                rr, cc = skimage.draw.polygon(seg[:,0], seg[:,1], mask_np.shape)
                mask_np[rr, cc, object_number] = 10 * object_number
                object_number += 1  # We are assigning each object a unique integer value (labeled mask)

    # Save the numpy array as a TIFF using tifffile library
    mask_path = os.path.join(
        output_folder, image_info["file_name"].replace(".*", "_mask.*")
    )
    #im = Image.fromarray(mask_np)
    #im.save(mask_path)
    #tifffile.imsave(mask_path, mask_np)
    imsave(mask_path, mask_np)

    print(f"Saved mask for {image_info['file_name']} to {mask_path}")

"""
----images
      [List of length 88 contains:]
--------file_name
--------height
--------width
--------id
----categories
      [List of length 18 contains:]
--------id
--------name
----annotations
      [List of length 305 contains:]
--------id
--------image_id
--------category_id
--------area
--------iscrowd
--------segmentation
          [List of length 1 contains:]
            [List of length 230 contains:]
--------bbox
          [List of length 4 contains:]
"""
def coco_mask(json_file, image_dir):
    coco = COCO(json_file)

    imgs = coco.getImgIds()
    num = len(imgs)
    print(f"num of images: {num}")

    for id in imgs:
        imgs = coco.loadImgs(ids=id)
        print(f"->>>{imgs}")
        
        cat_ids = coco.getCatIds()
        anns_ids =  coco.getAnnIds(imgIds=id, catIds=cat_ids, iscrowd=None)
        anns = coco.loadAnns(anns_ids)
        mask = coco.annToMask(anns[0])
        for i in range(len(anns)):
            mask += coco.annToMask(anns[i])
        imsave('./1.png', mask)



def main(json_file, mask_output_folder, image_output_folder, original_image_dir):
    # Load COCO JSON annotations
    with open(json_file, "r") as f:
        data = json.load(f)

    images = data["images"]
    annotations = data["annotations"]

    # Ensure the output directories exist
    if not os.path.exists(mask_output_folder):
        os.makedirs(mask_output_folder)
    if not os.path.exists(image_output_folder):
        os.makedirs(image_output_folder)

    for img in images:
        # Create the masks
        #create_mask(img, annotations, mask_output_folder)
        i_id = img['id']

        # Copy original images to the specified folder
        original_image_path = os.path.join(original_image_dir, img["file_name"])

        new_image_path = os.path.join(
            image_output_folder, os.path.basename(original_image_path)
        )
        shutil.copy2(original_image_path, new_image_path)
        print(f"Copied original image to {new_image_path}")


if __name__ == "__main__":
    original_image_dir = "/home/madhekar/work/vision/research/code/test/annotations/images"
    json_file = '/home/madhekar/work/vision/research/code/test/annotations/annotations.json'
    mask_output_folder = ("/home/madhekar/work/vision/research/code/test/annotations/val/masks")
    image_output_folder = "/home/madhekar/work/vision/research/code/test/annotations/val/images"  #
    #main(json_file, mask_output_folder, image_output_folder, original_image_dir)
    coco_mask(json_file, original_image_dir)
