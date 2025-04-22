import os
import torch, detectron2

TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print(torch.cuda.is_available())
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2:", detectron2.__version__)

from detectron2.utils.logger import setup_logger

setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.data.datasets import register_coco_instances

register_coco_instances(
    "zesha_dataset_train",
    {},
    "/home/madhekar/work/vision/research/code/test/annotations/annotations.json",
    "/home/madhekar/work/vision/research/code/test/annotations/images",
)
register_coco_instances(
    "zesha_dataset_val",
    {},
    "/home/madhekar/work/vision/research/code/test/annotations/annotations.json",
    "/home/madhekar/work/vision/research/code/test/annotations/images",
)

"""
 the metadata and dataset dictionaries for both training and validation datasets. 
 These can be used later for other purposes, like visualization, model training, evaluation, etc.
"""
train_metadata = MetadataCatalog.get("zesha_dataset_train")
train_dataset_dicts = DatasetCatalog.get("zesha_dataset_train")
val_metadata = MetadataCatalog.get("zesha_dataset_val")
val_dataset_dicts = DatasetCatalog.get("zesha_dataset_val")

cfg = get_cfg()
cfg.MODEL.DEVICE = "cpu"
cfg.OUTPUT_DIR = "/home/madhekar/work/home-media-app/models/detectron2"
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("zesha_dataset_train",)
cfg.DATASETS.TEST = ("zesha_dataset_train",)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 17 

# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2  # set a custom testing threshold
trainer = DefaultTrainer(cfg)
predictor = DefaultPredictor(cfg)     

trainer.resume_or_load(resume=False)

from detectron2.utils.visualizer import ColorMode
from matplotlib import pyplot as plt

for d in random.sample(val_dataset_dicts, 1):  # select number of images for display
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    print(f'{outputs["instances"].pred_boxes} : {d["file_name"]}')
    v = Visualizer(
        im[:, :, ::-1],
        metadata=val_metadata,
        scale=0.5,
        instance_mode=ColorMode.IMAGE_BW,  # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    vis = v.draw_dataset_dict(d)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.imshow( out.get_image()[:, :, ::-1])
    plt.show()


# from detectron2.evaluation import COCOEvaluator, inference_on_dataset
# from detectron2.data import build_detection_test_loader

# evaluator = COCOEvaluator("zesha_dataset_val", output_dir="./output")
# val_loader = build_detection_test_loader(cfg, "zesha_dataset_val")
# print(inference_on_dataset(predictor.model, val_loader, evaluator))
# another equivalent way to evaluate the model is to use `trainer.test`

# raw_image = '/home/madhekar/work/home-media-app/data/input-data/img/IMG-20190706-WA0000.jpg'
# im = cv2.imread(raw_image)
# op = predictor(im)
# print(op)
# v = Visualizer(im[:, :, ::-1], metadata= train_metadata, scale=.5, instance_mode=ColorMode.IMAGE_BW,)
# out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
# plt.imshow( out.get_image()[:, :, ::-1])
# plt.show()


import csv
from skimage.measure import regionprops, label


# Assuming you have already defined the 'predictor' object and loaded the model.
# Also, make sure 'metadata' is defined appropriately.

# Directory path to the input images folder
input_images_directory = "/home/madhekar/work/home-media-app/data/input-data/img/"

# Output directory where the CSV file will be saved
output_csv_path = "./output_objects.csv"  # Replace this with the path to your desired output CSV file

# Open the CSV file for writing
with open(output_csv_path, "w", newline="") as csvfile:
    csvwriter = csv.writer(csvfile)

    # Write the header row in the CSV file
    csvwriter.writerow(
        ["File Name", "Class Labels", "Class Names"]#, "Object Number", "Area", "Centroid", "BoundingBox"]
    )  # Add more columns as needed for other properties

    # Loop over the images in the input folder
    for image_filename in os.listdir(input_images_directory):
        image_path = os.path.join(input_images_directory, image_filename)
        new_im = cv2.imread(image_path)

        # Perform prediction on the new image
        outputs = predictor(
            new_im
        )  # Format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format

        # Convert the predicted mask to a binary mask
        mask = outputs["instances"].pred_masks.to("cpu").numpy().astype(bool)

        # Get the predicted class labels
        class_labels = outputs["instances"].pred_classes.to("cpu").numpy()

        pred_classes = outputs["instances"].pred_classes.cpu().tolist()
        class_names = train_metadata.thing_classes
        pred_class_names = list(map(lambda x: class_names[x], pred_classes))

        # Debugging: print class_labels and metadata.thing_classes
        print("File Name:", image_filename,  "Class Labels:", class_labels, "Class Names", pred_class_names)
        #print("Thing Classes:", train_metadata.thing_classes)

        

        # Use skimage.measure.regionprops to calculate object parameters
        # labeled_mask = label(mask)
        # props = regionprops(labeled_mask)

        # # Write the object-level information to the CSV file
        # for i, prop in enumerate(props):
        #     # object_number = i + 1  # Object number starts from 1
        #     # area = prop.area
        #     # centroid = prop.centroid
        #     # bounding_box = prop.bbox

        #     # Check if the corresponding class label exists
        #     if i < len(class_labels):
        #         class_label = class_labels[i]
        #         class_name = train_metadata.thing_classes[class_label]
        #         print(f'{i} : {class_label} : {class_name}')
        #     else:
        #         # If class label is not available (should not happen), use 'Unknown' as class name
        #         class_name = "Unknown"

        # Write the object-level information to the CSV file
        csvwriter.writerow(
            [
                image_filename,
                class_labels,
                pred_class_names,
                # object_number,
                # area,
                # centroid,
                # bounding_box,
            ]
        )  # Add more columns as needed for other properties

print("Object-level information saved to CSV file.")