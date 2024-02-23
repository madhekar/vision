import streamlit as st
import torch,torchvision
# import req python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

from detectron2.utils.logger import setup_logger
import numpy as np
import os, json, cv2, random
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

print('Torch version: ' ,torch.__version__, 'Is GPU available?: ', torch.cuda.is_available())

setup_logger()

# import some common detectron2 utilities
st.title('Zesha Detectron2 POC')
st.write('Test image for messy - home kitched')

# load zesha sample image for detection
im = cv2.imread("assets/kitchen_mess.jpg")

# raw image display using streamlit screen
st.image('assets/kitchen_mess.jpg')

#Then, we create a detectron2 config and a detectron2 `DefaultPredictor` to run inference on this image.
cfg = get_cfg()

# check if cuda support is available
if torch.cuda.is_available():
  cfg.MODEL.DEVICE = "cuda"
else:
  cfg.MODEL.DEVICE = "cpu"


# add specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

# set threshold for this model
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 

# model from detectron2's model zoo. 
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

# print('detectron2 configuration:', cfg)

# get detectron2 predictor
predictor = DefaultPredictor(cfg)

# predicted outputs
outputs = predictor(im)

# Writing pred_classes/pred_boxes output
st.write('Writing pred_classes/pred_boxes output')
st.write(outputs["instances"].pred_classes)
st.write(outputs["instances"].pred_boxes)


st.write('Using Vizualizer to draw the predictions on Image')
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)

# apply outputs on top of image
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

# display image on screen
st.image(out.get_image()[:, :, ::-1])