
import os
import shutil
from pycocotools.coco import COCO
from matplotlib import pyplot as plt

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
def coco_mask(json_file, mask_output_folder, image_output_folder, original_image_dir):
    coco = COCO(json_file)

    # Ensure the output directories exist
    if not os.path.exists(mask_output_folder):
        os.makedirs(mask_output_folder)
    if not os.path.exists(image_output_folder):
        os.makedirs(image_output_folder)

    imgs = coco.getImgIds()
    cat_ids = coco.getCatIds()

    for id in imgs:

        imgs = coco.imgs[id]
        #print(f"->>>{imgs}")
        
        anns_ids =  coco.getAnnIds(imgIds=id, catIds=cat_ids, iscrowd=None)
        anns = coco.loadAnns(anns_ids)
        #print(anns)
        mask = coco.annToMask(anns[0])
        for i in range(len(anns)):
            mask += coco.annToMask(anns[i])
        #imsave(f'{cnt}.png', mask)
        print(len(mask))

        # Copy original images to the specified folder
        original_image_path = os.path.join(original_image_dir, imgs["file_name"])

        new_image_path = os.path.join(
            image_output_folder, os.path.basename(original_image_path)
        )
        shutil.copy2(original_image_path, new_image_path)
        print(f"Copied original image to {new_image_path}")

        fn = imgs['file_name'].split('.')

        fn1 = ''.join(fn[:-1]) + '_mask.'+ fn[-1]
        mask_path = os.path.join(mask_output_folder, fn1)
                                 
        plt.imsave(mask_path, mask)                         

if __name__ == "__main__":
    original_image_dir = "/home/madhekar/work/vision/research/code/test/annotations/images"
    json_file = '/home/madhekar/work/vision/research/code/test/annotations/annotations.json'
    mask_output_folder = ("/home/madhekar/work/vision/research/code/test/annotations/val/masks")
    image_output_folder = "/home/madhekar/work/vision/research/code/test/annotations/val/images"  #
    #main(json_file, mask_output_folder, image_output_folder, original_image_dir)
    coco_mask(json_file, mask_output_folder, image_output_folder, original_image_dir)
