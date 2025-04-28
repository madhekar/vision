import os
import random
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

'''
annotations/annotations.json
'''

def print_structure(d, ident=2):

   if isinstance(d, dict):
       for k, v in d.items():
          print('--'* ident + str(k))
          print_structure(v, ident+1)

   if isinstance(d, list):
      print('  ' * ident + "[List of length {} contains:]".format(len(d)))
      if d:
         print_structure(d[0], ident+1)



def display_images_with_coco_annotations(
    image_paths, annotations, display_type="both", colors=None
):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    for ax, img_path in zip(axs.ravel(), image_paths):
        # Load image using OpenCV and convert it from BGR to RGB color space
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ax.imshow(image)
        ax.axis("off")  # Turn off the axes

        # Define a default color map if none is provided
        if colors is None:
            colors = plt.get_cmap("tab10")

        # Get image filename to match with annotations
        img_filename = os.path.basename(img_path)
        img_id = next(
            item for item in annotations["images"] if item["file_name"] == img_filename
        )["id"]

        # Filter annotations for the current image
        img_annotations = [
            ann for ann in annotations["annotations"] if ann["image_id"] == img_id
        ]

        for ann in img_annotations:
            category_id = ann["category_id"]
            color = colors(category_id % 10)

            # Display bounding box
            if display_type in ["bbox", "both"]:
                bbox = ann["bbox"]
                rect = patches.Rectangle(
                    (bbox[0], bbox[1]),
                    bbox[2],
                    bbox[3],
                    linewidth=1,
                    edgecolor=color,
                    facecolor="none",
                )
                ax.add_patch(rect)

            # Display segmentation polygon
            if display_type in ["seg", "both"]:
                for seg in ann["segmentation"]:
                    poly = [(seg[i], seg[i + 1]) for i in range(0, len(seg), 2)]
                    polygon = patches.Polygon(
                        poly, closed=True, edgecolor=color, fill=False
                    )
                    ax.add_patch(polygon)

    plt.tight_layout()
    plt.show()


with open('/home/madhekar/work/vision/research/code/test/annotations/annotations.json') as file:
    data = json.load(file)

    # for img in data['images'][:10]:
    #     print(img['file_name'])

    #print_structure(data)

    image_dir = '/home/madhekar/work/vision/research/code/test/annotations/images'

    all_image_files = [
        os.path.join(image_dir, img["file_name"]) for img in data["images"]
    ]
    random_image_files = random.sample(all_image_files, 4)

    # Choose between 'bbox', 'seg', or 'both'
    display_type = "seg"
    display_images_with_coco_annotations(random_image_files, data, display_type)