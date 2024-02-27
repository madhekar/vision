import time
import matplotlib.pyplot as plt 
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision
import numpy as np 
import cv2
import streamlit as st
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
        
    def _display_detected_frames(self, conf, model, st_frame, image, is_display_tracking=None, tracker=None):
        """
            Display the detected objects on a video frame using the FASTRCNN model.

            Args:
                - conf (float): Confidence threshold for object detection.
                - model (FASTRCNN): 
                - st_frame (Streamlit object): A Streamlit object to display the detected video.
                - image (numpy array): A numpy array representing the video frame.
                - is_display_tracking (bool): A flag indicating whether to display object tracking (default=None).

            Returns:
                None
        """
        # Resize the image to a standard size
        image = cv2.resize(image, (720, int(720*(9/16))))
        st_frame.image(image,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )

    def initEnv(self):
        st.title('Zesha Detectron2 POC')
        st.write('Torch version: ' ,torch.__version__, 'Is GPU available?: ', torch.cuda.is_available())
        st.write('Test image:  messy - home kitched')
        # todo check cuda support
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT) 
        self.model.eval()

    def getPrediction(self, imgPath, threshold):
        img = Image.open(imgPath)
        transform = T.Compose([T.ToTensor()])
        img = transform(img)
        pred = self.model([img])
        
        pred_score = list(pred[0]['scores'].detach().cpu().numpy())
        pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]

        pred_class = [self.COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]

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
          - for each prediction, bounding box is drawn and text is written with opencv
          - the final image is displayed
        """
        boxes, pred_cls = self.getPrediction(imgPath, threshold)

        img = cv2.imread(imgPath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for i in range(len(boxes)):
           pt1 = tuple(int(x) for x in boxes[i][0])
           pt2 = tuple(int(x) for x in boxes[i][1])
           cv2.rectangle(img, pt1, pt2, color=(0, 255, 0), thickness=rect_th)
           cv2.putText(img,pred_cls[i], pt1, cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
        st_frame = st.empty()
        self._display_detected_frames(0.5, 'fastRCNN', st_frame, img, False, None) 

    def getImagePrediction(self, img, threshold):

        transform = T.Compose([T.ToTensor()])
        plt.imshow(img)
        img = transform(img)
        pred = self.model([img])
        print(pred)
        pred_score = list(pred[0]['scores'].detach().cpu().numpy())
        print(pred_score)
        pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]

        

        pred_class = [self.COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]

        pred_boxes = pred_boxes[:pred_t+1]
        pred_class = pred_class[:pred_t+1]
        print(pred_boxes, pred_class)
        return pred_boxes, pred_class


    def detectEntitiesVideo(self, threshold=0.5, rect_th=1, text_size=.5, text_th=1):

        source_webcom = 0
        try:
            vid_cap = cv2.VideoCapture(source_webcom)

  
            while (vid_cap.isOpened()):
              
                time.sleep(10)

                success, img = vid_cap.read()

                if success:
                     
                     img = cv2.resize(img,(720, int(720*(9/16))))

                     boxes, pred_cls = self.getImagePrediction(img, threshold)
                     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                     
                     
                     for i in range(len(boxes)):
                       pt1 = tuple(int(x) for x in boxes[i][0])
                       pt2 = tuple(int(x) for x in boxes[i][1])
                       cv2.rectangle(img, pt1, pt2, color=(0, 255, 0), thickness=rect_th)
                       cv2.putText(img,pred_cls[i], pt1, cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
   
                     st_frame = st.empty()
                     self._display_detected_frames(0.5, 'fastRCNN', st_frame, img, False, None) 
                else:
                    vid_cap.release()
                    break

        except Exception as e:
            st.sidebar.error('error in read/ load/ process video: ' + str(e))             
                

if __name__=='__main__':
    zd = zsehaDetector()

    zd.initEnv()    

    #zd.dectectEntities('./messy_kitchen.jpg', threshold=.5)

    zd.detectEntitiesVideo(threshold=0.5)
