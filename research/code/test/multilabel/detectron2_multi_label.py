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
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


im = cv2.imread(
    "/home/madhekar/work/home-media-app/data/train-data/img/AnjaliBackup/IMAG2400.jpg"
)
cv2.imshow('sample', im)


cfg = get_cfg()
cfg.MODEL.DEVICE = "cpu"
print(cfg)
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo.  https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
)
predictor = DefaultPredictor(cfg)
outputs = predictor(im)

# look at the outputs - tensors and bounding boxes.
print(outputs["instances"].pred_classes)
print(outputs["instances"].pred_boxes)

# We can use `Visualizer` to draw the predictions on the image.
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=0.8)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imshow('viz', out.get_image()[:, :, ::-1])


'''
Import the necessary function to register datasets in the COCO format. Let us register both the training and validation datasets. 
Please note that we are working with training (and validation) data that is is the coco format where we have a single JSON file that 
describes all the annotations from all training images.
'''
from detectron2.data.datasets import register_coco_instances

register_coco_instances(
    "zesha_dataset_train",
    {},
    "/home/madhekar/work/home-media-app/data/train-data/img/AnjaliBackup/data.json",
    "/home/madhekar/work/home-media-app/data/train-data/img/AnjaliBackup",
)
register_coco_instances(
    "zesha_dataset_val",
    {},
    "/home/madhekar/work/home-media-app/data/train-data/img/AnjaliBackup/data.json",
    "/home/madhekar/work/home-media-app/data/train-data/img/AnjaliBackup",
)

"""
 the metadata and dataset dictionaries for both training and validation datasets. 
 These can be used later for other purposes, like visualization, model training, evaluation, etc.
"""
train_metadata = MetadataCatalog.get("zesha_dataset_train")
train_dataset_dicts = DatasetCatalog.get("zesha_dataset_train")
val_metadata = MetadataCatalog.get("zesha_dataset_val")
val_dataset_dicts = DatasetCatalog.get("zesha_dataset_val")


from matplotlib import pyplot as plt

print(train_dataset_dicts)

# Visualize some random samples
for d in random.sample(train_dataset_dicts,1):
    print(f'-->{d}')
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=train_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    plt.imshow(vis.get_image()[:, :, ::-1])
    plt.show()

"""
to train a Mask R-CNN model using the Detectron2 library. 
setting up a configuration file (.cfg) for the model. 
The configuration file contains many details including the output directory path, 
training dataset information, pre-trained weights, base learning rate, maximum number of iterations, etc.
"""

from detectron2.engine import DefaultTrainer

cfg = get_cfg()
cfg.MODEL.DEVICE = "cpu"
cfg.OUTPUT_DIR = "/home/madhekar/work/home-media-app/models/detectron2"
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
)
cfg.DATASETS.TRAIN = ("zesha_dataset_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
)  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = (
    2  # This is the real "batch size" commonly known to deep learning people
)
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 1000  # 1000 iterations seems good enough for this dataset
cfg.SOLVER.STEPS = []  # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
    512  # Default is 512, using 256 for this dataset.
)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 18  # We have 4 classes.
# NOTE: this config means the number of classes, without the background. Do not use num_classes+1 here.

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(
    cfg
)  # Create an instance of of DefaultTrainer with the given congiguration
trainer.resume_or_load(
    resume=False
)  # Load a pretrained model if available (resume training) or start training from scratch if no pretrained model is available

# train 

trainer.train()  # Start the training process