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
import os,torchvision
import numpy as np
from PIL import Image
import collections
import torchvision.transforms as transforms
# from segmentation_models_pytorch import Unet
from segment_anything import build_sam, SamAutomaticMaskGenerator,sam_model_registry, SamPredictor
from huggingface_hub import hf_hub_download
import torch


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


        DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        MODEL_TYPE = "vit_h"
        CHECKPOINT_PATH = 'sam_vit_h_4b8939.pth'

        sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
        sam.to(device=DEVICE)

        # mask_generator = SamAutomaticMaskGenerator(sam)
        mask_generator = SamPredictor(sam)
        return mask_generator , yolo_model
    
    def read_img(self,uploaded_file):
      image = Image.open(uploaded_file).convert('RGB')
      image = np.array(image)
      return image
    
    def inference(self,image):
        mask_predictor , yolo_model = self.model
        yolo_results  =yolo_model(image,size = 640)
        x_min,y_min,x_max,y_max,confi,cla = yolo_results.xyxy[0][0].numpy()
        mask_predictor.set_image(image)
        st.text(yolo_results.pandas().xyxy)
        masks, scores, logits = mask_predictor.predict(
            box=np.array([x_min,y_min,x_max,y_max]),
            multimask_output=False
        )
        return masks
      
    def render(self):
        st.title("Segmentation")
        uploaded_file = st.file_uploader("Choose a file")

        if uploaded_file is not None:
          image = st.image(uploaded_file,use_column_width=True)
          image = self.read_img(uploaded_file)
          st.text(image.shape)
          masks = self.inference(image)
          st.text(len(masks))
          st.image(masks[0].astype(float))
        
app = L.LightningApp(StreamlitApp())
