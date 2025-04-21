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

for d in random.sample(val_dataset_dicts, 5):  # select number of images for display
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
    #out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.imshow( vis.get_image()[:, :, ::-1])
    plt.show()


# from detectron2.evaluation import COCOEvaluator, inference_on_dataset
# from detectron2.data import build_detection_test_loader

# evaluator = COCOEvaluator("zesha_dataset_val", output_dir="./output")
# val_loader = build_detection_test_loader(cfg, "zesha_dataset_val")
# print(inference_on_dataset(predictor.model, val_loader, evaluator))
# another equivalent way to evaluate the model is to use `trainer.test`
