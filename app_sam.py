# app.py
# !pip install streamlit omegaconf scipy
# !pip install torch
import lightning as L
import torch
from io import BytesIO
from functools import partial
import streamlit as st
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from PIL import Image
# from segmentation_models_pytorch import Unet
from segment_anything import  SamAutomaticMaskGenerator,sam_model_registry, SamPredictor
import torch
import cv2
from skimage.morphology import skeletonize
from sklearn.metrics.pairwise import euclidean_distances
from streamlit_drawable_canvas import st_canvas
import pandas as pd

class StreamlitApp(L.app.components.ServeStreamlit):
    def show_anns(self, anns):

        if len(anns) == 0:
            return
        
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        for ann in sorted_anns:
            m = ann['segmentation']
            img = np.ones((m.shape[0], m.shape[1], 3))
            color_mask = np.random.random((1, 3)).tolist()[0]
            for i in range(3):
                img[:,:,i] = color_mask[i]

            return img
        # return img #np.dstack((img, m*0.35))
            
    def build_model(self):
        # detection model
        yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

        keypoint_yolo = torch.hub.load('ultralytics/yolov5', 'custom', path='keypoint_best.pt')

        DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        MODEL_TYPE = "vit_h"
        CHECKPOINT_PATH = 'sam_vit_h_4b8939.pth'

        sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
        sam.to(device=DEVICE)

        # mask_generator = SamAutomaticMaskGenerator(sam)
        mask_generator = SamPredictor(sam)
        return mask_generator , yolo_model , keypoint_yolo
    
    def read_img(self,uploaded_file):
      image = Image.open(uploaded_file).convert('RGB')
      image = np.array(image)
      return image
    def get_centers(self,df):
        centers, classes = [],[]
        for idx , row in df.iterrows():
            centers.append((int((row['xmax'] + row['xmin'])/2), int((row['ymax'] + row['ymin'])/2)))
            classes.append(row['class'])

        return centers,classes

    def detection_op(self,yolo_results, yolo_keypoints_results,image):

        st.write(yolo_results.pandas().xyxy[0])
        st.write(yolo_keypoints_results.pandas().xyxy[0])
        centers,classes = self.get_centers(yolo_keypoints_results.pandas().xyxy[0])
        x_min,y_min,x_max,y_max,confi,cla = yolo_results.xyxy[0][0].numpy()
        
        color = (255, 0, 0)
        # Line thickness of 2 px
        thickness = 2
        img_copy = np.copy(image)
        for point in centers:
            cv2.circle(img_copy, tuple(point), 1, (255,0,0),10)
        cv2.rectangle(img_copy,[int(x_min),int(y_min)],[int(x_max),int(y_max)],color, thickness)
        st.image(img_copy) 

        return x_min,y_min,x_max,y_max
    

    def render(self):
        st.title("Morphology Assistant")
        uploaded_file = st.file_uploader("Choose a file")

        if uploaded_file is not None:
          
          image = st.image(uploaded_file,use_column_width=True)
          image = self.read_img(uploaded_file)

          st.text(f"Image shape is : {image.shape}")
          mask_predictor , yolo_model , keypoint_yolo = self.model
          yolo_results  =yolo_model(image,size = 640)
          yolo_keypoints_results = keypoint_yolo(image, size= 640)

          x_min,y_min,x_max,y_max = self.detection_op(yolo_results,yolo_keypoints_results,image)
          mask_predictor.set_image(image)

          masks, scores, logits = mask_predictor.predict(
            box=np.array([x_min,y_min,x_max,y_max]),
            multimask_output=False
                )
          mask = masks[0].astype(float)
          st.image(mask)
          pixels = cv2.countNonZero(mask)
          st.text(f"pixels covered by seleted area {pixels}"
            )
        #   ret, thresh = cv2.threshold(mask, 127, 255, 0)
          contours, _ = cv2.findContours(np.array(mask, np.uint8), cv2.RETR_TREE,
                               cv2.CHAIN_APPROX_SIMPLE)
          cnt = max(contours, key = cv2.contourArea)
          
          def midpoint(ptA, ptB): return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

          # take the first contour
          rect = cv2.minAreaRect(cnt)
          box = cv2.boxPoints(rect)
          box = np.int0(box)
        # Line thickness of 2 px
          (tl, tr, br, bl) = box

          (tltrX, tltrY) = midpoint(tl,tr)
          (blbrX, blbrY) = midpoint(bl,br)
          (tlblX, tlblY) = midpoint(tl, bl)
          (trbrX, trbrY) = midpoint(tr, br)

          distance1 = np.sqrt(np.sum(np.square(np.array([int(tlblX), int(tlblY)])-  np.array([int(trbrX), int(trbrY)]))))
          distance2 = np.sqrt(np.sum(np.square(np.array([int(tltrX), int(tltrY)])-  np.array([int(blbrX), int(blbrY)]))))
          cv2.drawContours(image, [box], 0, (0, 0, 255), 2)
          if distance1>=distance2:
              cv2.line(image, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),(255, 0, 255), 2)
              st.text(f'length in pixels : euclidian(p1,p2) {distance1}')
          else :
              cv2.line(image, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),(255, 0, 255), 2)
              st.text(f'length in pixels : euclidian(p1,p2) {distance2}')

          st.image(image)
          
        
app = L.LightningApp(StreamlitApp())
