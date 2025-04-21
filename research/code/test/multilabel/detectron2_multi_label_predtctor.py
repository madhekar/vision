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


cfg = get_cfg()
cfg.MODEL.DEVICE = "cpu"
cfg.OUTPUT_DIR = "/home/madhekar/work/home-media-app/models/detectron2"
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
# cfg.DATASETS.TRAIN = ("zesha_dataset_train",)
# cfg.DATASETS.TEST = ()
# cfg.DATALOADER.NUM_WORKERS = 2
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
# cfg.SOLVER.IMS_PER_BATCH = (2) # This is the real "batch size" commonly known to deep learning people
# cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
# cfg.SOLVER.MAX_ITER = 1000  # 1000 iterations seems good enough for this dataset
# cfg.SOLVER.STEPS = []  # do not decay learning rate
# cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (256)# Default is 512, using 256 for this dataset.
# cfg.MODEL.ROI_HEADS.NUM_CLASSES = 17  # We have 4 classes.

# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set a custom testing threshold
predictor = DefaultPredictor(cfg)     

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

from detectron2.utils.visualizer import ColorMode

for d in random.sample(val_dataset_dicts, 10):  # select number of images for display
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    print(f'{outputs["instances"]} : {d["file_name"]}')
    v = Visualizer(
        im[:, :, ::-1],
        metadata=val_metadata,
        scale=0.5,
        instance_mode=ColorMode.IMAGE_BW,  # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow('QA', out.get_image()[:, :, ::-1])


from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

evaluator = COCOEvaluator("zesha_dataset_val", output_dir="./output")
val_loader = build_detection_test_loader(cfg, "zesha_dataset_val")
print(inference_on_dataset(predictor.model, val_loader, evaluator))
# another equivalent way to evaluate the model is to use `trainer.test`
