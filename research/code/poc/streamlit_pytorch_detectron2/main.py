import numpy as np
import os, json, cv2, random
import torch,torchvision
import streamlit as st

# import req python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.logger import setup_logger

class zeshDetector:
  def __init__(self):
    self.cfg = get_cfg()
    self.detector = None
    self.outputs = None

  def initEnv(self):
     st.title('Zesha Detectron2 POC')
     setup_logger()
     print('Torch version: ' ,torch.__version__, 'Is GPU available?: ', torch.cuda.is_available())
     st.write('Test image for messy - home kitched')
     # check if cuda support is available
     if torch.cuda.is_available():
        self.cfg.MODEL.DEVICE = "cuda"
     else:
        self.cfg.MODEL.DEVICE = "cpu"

  def initModel(self, threshold):
     self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

     # set threshold for this model
     self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold #0.5 

     # model from detectron2's model zoo. 
     self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

     self.detector = DefaultPredictor(self.cfg)

  def genDetectorOutput(self, image):
        self.outputs = self.detector(image)
        st.write('Writing pred_classes/pred_boxes output')
        st.write(self.outputs["instances"].pred_classes)
        st.write(self.outputs["instances"].pred_boxes)
     
  def visualize(self, image):
     st.write('Using Vizualizer to draw the predictions on Image')
     v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)

     # apply outputs on top of image
     out = v.draw_instance_predictions(self.outputs["instances"].to("cpu"))

     # display image on screen
     st.image(out.get_image()[:, :, ::-1])   


if __name__ == '__main__':
   zd = zeshDetector()

   zd.initEnv()

   zd.initModel(0.5)

   # load zesha sample image for detection
   im = cv2.imread("assets/kitchen_mess.jpg")

   # raw image display using streamlit screen
   st.image('assets/kitchen_mess.jpg')

   zd.genDetectorOutput(im)

   zd.visualize(im)
