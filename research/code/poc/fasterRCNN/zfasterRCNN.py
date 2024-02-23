import matplotlib.pyplot as plt 
from PIL import Image
import random
import torch
import torchvision.transforms as T
import torchvision
import numpy as np 
import cv2
import warnings

class zsehaDetector:
    def __init__(self):
        self.model = None
        self.COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    def initEnv(self):
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT) 
        self.model.eval()

    def getPrediction(self, imgPath, threshold):
        img = Image.open(imgPath)
        transform = T.Compose([T.ToTensor()])
        img = transform(img)
        pred = self.model([img])
        
        pred_score = list(pred[0]['scores'].detach().cpu().numpy())
        pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
        #masks = (pred[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()
        pred_class = [self.COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
        #masks = masks[:pred_t+1]
        pred_boxes = pred_boxes[:pred_t+1]
        pred_class = pred_class[:pred_t+1]
        return pred_boxes, pred_class
    
    
    def dectectEntities(self, imgPath, threshold=0.5, rect_th=1, text_size=.5, text_th=1):
        """
        parameters:
          - img_path - path of the input image
          - confidence - threshold value for prediction score
          - rect_th - thickness of bounding box
          - text_size - size of the class label text
          - text_th - thichness of the text
        method:
          - prediction is obtained from get_prediction method
          - for each prediction, bounding box is drawn and text is written 
            with opencv
          - the final image is displayed
        """
        boxes, pred_cls = self.getPrediction(imgPath, threshold)

        img = cv2.imread(imgPath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # print(len(boxes))
        for i in range(len(boxes)):
           pt1 = tuple(int(x) for x in boxes[i][0])
           pt2 = tuple(int(x) for x in boxes[i][1])
           cv2.rectangle(img, pt1, pt2, color=(0, 255, 0), thickness=rect_th)
           cv2.putText(img,pred_cls[i], pt1, cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
        #plt.figure(figsize=(10,10))
        plt.imshow(img)
        #plt.xticks([])
        #plt.yticks([])
        plt.show()
    
if __name__=='__main__':
    zd = zsehaDetector()

    zd.initEnv()    

    zd.dectectEntities('./messy_kitchen.jpg', threshold=.5)