# app.py
# !pip install streamlit omegaconf scipy
# !pip install torch
import lightning as L
import torch
# from io import BytesIO
# from functools import partial
from scipy.io.wavfile import write
import streamlit as st
import warnings
warnings.filterwarnings('ignore')
import os,torchvision
import numpy as np
from PIL import Image
import collections
import torchvision.transforms as transforms
from segmentation_models_pytorch import Unet

class StreamlitApp(L.app.components.ServeStreamlit):
    def build_model(self):
        CKPT_PATH = '/Users/sagar/Desktop/AI_cap/segmentation/super-waddle/lightning_logs/version_67/checkpoints/epoch=31-step=224.ckpt'
        chkpt = torch.load(CKPT_PATH,map_location=torch.device('cpu'))

        # test_folder = '/Users/sagar/Desktop/AI_cap/sturdy-eureka/data/yolo_training_data/images'
        # img_paths =[ os.path.join(test_folder,image_name) for image_name in os.listdir(test_folder)]
        model = model =  Unet(
                                encoder_name='efficientnet-b3',        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                                encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                                in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                                activation='sigmoid',
                        )

        state_dict_new = collections.OrderedDict()
        for key,val in chkpt["state_dict"].items():
            key = key[6:]
            state_dict_new[key]=val

        model.load_state_dict(state_dict_new)
        model.eval()
        return model
    

    def read_img(self,uploaded_file):
      image = Image.open(uploaded_file)
      transform = transforms.Compose([
                                transforms.Resize(size = (640,640)),
                                transforms.PILToTensor()
                            ])
      image = transform(image)
      image = image/255.
      image = torch.unsqueeze(image, dim=0)
      return image
      
    def render(self):
        st.title("Segmentation")
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
          image = st.image(uploaded_file,use_column_width=True)
          image = Image.open(uploaded_file)
          image = self.read_img(uploaded_file)
          preds = (self.model(image) > 0.75).numpy().astype(float)
          st.image(image[0].permute(1,2,0).numpy())
          st.text(preds[0][0])
          st.image(preds[0][0])
        
app = L.LightningApp(StreamlitApp())
