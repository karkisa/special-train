import pandas as pd
import torch
import cv2,os
import numpy as np
from PIL import Image
import bezier
from scipy.special import comb
from segment_anything import  sam_model_registry, SamPredictor
import collections,math
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog,messagebox
from glob import glob 
import os
import lightning as L
import lightning_app.frontend as frontend
import streamlit as st
import cv2, pandas as pd
import lightning_app as app       
import torch

def get_centers(df):
    conf = [-1]*6
    fins = []
    centers = collections.defaultdict()
    mark_idx = 0

    mask_labels = []
    
    for idx , row in df.iterrows():
        key_pt = [int((row['xmax'] + row['xmin'])/2), int((row['ymax'] + row['ymin'])/2)]
        
        if row['class']==6:
            fins.append(key_pt)

        else:
            if row["confidence"]>conf[row['class']] :
                mark_idx+=1
                centers[row['class']]=key_pt
                conf[row['class']]  = row["confidence"]

    center_pts = []
    for i in range(0,6):
        if i in centers:
            center_pts.append(centers[i])
            mask_labels.append(1)  # 1 vor visibilty

    for pts in fins:
        center_pts.append(pts)
        mask_labels.append(0)   # point not included in mask

    return np.array(center_pts),np.array(mask_labels),mark_idx,centers

def get_bezire_curve(centers):
        
        nodes1 = np.asfortranarray([
            centers[:,0], centers[:,1]
            ])
        # print(nodes1)
        curve1 = bezier.Curve(nodes1, degree=len(centers)-1)

        return curve1

def B(i, N, t):
        val = comb(N,i) * t**i * (1.-t)**(N-i)
        return val

def P(t, X):
     
        X = np.array(X)
        N,d = np.shape(X)   # Number of points, Dimension of points
        N = N - 1
        xx = np.zeros((len(t), d))
        
        for i in range(N+1):
            xx += np.outer(B(i, N, t), X[i])
        return xx 

def get_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    
    return image

def get_pixel_length(image,centers):
  
    
    if len(centers)==0:
         return 0,image.shape[1]
    print(centers)
    curve = get_bezire_curve(centers)

    return curve,image.shape[1]

def create_binary_mask(polygon, width, height):
    
    # Create an empty mask
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Fill the polygon in the mask with white (255)
    cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], 255)
    
    return mask

def get_mask_yolov8(model,image_path):
     results = model(image_path)
     height,width,channels =get_image(image_path).shape
     polygon = results[0].masks.xy[0]  

     mask = create_binary_mask(polygon, width, height)
     
     return mask,polygon

def get_roi(mask_predictor,image,centers,mask_labels):
          mask_predictor.set_image(image)
          masks, scores, logits = mask_predictor.predict(
            # box=box_cords,
            point_coords=centers,
            point_labels=mask_labels,
            multimask_output=False
                )
          mask = masks[0].astype(float)
          return  mask

def get_dir(p1,p2):
    if p1<p2:
        return 1
    return -1

def remove_mask_quads(curve,image,mask,start,end):

    x1,y1 = curve.evaluate(start)
    x2,y2 =  curve.evaluate(end)

    pt_set1 = get_quadpts(start,0.0,curve,image,mask)
    pt_set2 = get_quadpts(end,1.0,curve,image,mask)

    pixels = cv2.countNonZero(mask)

    return pixels, (end-start)

def save_img(path,image,mask,centers,base_save_folder):
    # pdb.set_trace()
    name = path.split('/')[-1]
    mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)[1]
    image[mask==255] = (0,255,0)
    for point in centers: 
            cv2.circle(image, tuple(point), 1, (255,0,0),10)
    cv2.imwrite(os.path.join(base_save_folder,name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def get_quadpts( percent,end_pt_ext,curve,image,mask):
    direction = -1 if end_pt_ext==0 else 1
    box_d = 0.15*curve.length
    extra_length = direction*0.2*curve.length

    vx,vy =  curve.evaluate_hodograph(percent)
    x1,y1 = curve.evaluate(percent)
    # clockwise - > [1,-1]
    pt1 = get_point_k_dist(vy,-vx,[x1,y1],box_d)
    pt2 = get_point_k_dist(-vy,vx,[x1,y1],box_d)
    

    x00,y00 = curve.evaluate(end_pt_ext)
    vx,vy =  curve.evaluate_hodograph(end_pt_ext)
    
    ext_pt = get_point_k_dist(vx,vy,[x00,y00],extra_length)

    pt3 = get_point_k_dist(-vy,vx,ext_pt,box_d)
    pt4 = get_point_k_dist(vy,-vx,ext_pt,box_d)


    points = np.array([pt1,pt2,pt3,pt4])
    
    cv2.fillPoly(mask,pts=np.int32([points]), color=0)
    
    return [pt1,pt2,pt3,pt4]

def get_point_k_dist(vx,vy,pt,d):
    x1,y1 = pt
    
    magnitude = math.sqrt(vx**2 + vy**2)
    x2 = x1 + (vx/magnitude) * d
    y2 = y1 + (vy/magnitude) * d

    return [int(x2),int(y2)]

def get_alt(image_path):
     image_name = image_path.split('/')[-1]
     alt = image_name.split('_000_')[-1]
     alt = alt.split('.')[0]
     alt = alt.split('_')

     try :
        alt = float(alt[0])+float(alt[1])/(10*len(alt[1]))
     except:
        alt = 50

     return alt

def analyse_csv(df,base_folder,sensor_width,key_point_model,yolo_seg_model,sam_mask_predictor,save_folder,start,end):
     
     for idx,row in df.iterrows():
          
          image_path = row["image"]
          if not os.path.exists(image_path):
               continue
          image = get_image(image_path)
          
          altitude_from_image = get_alt(image_path)
          altitude = altitude_from_image +row['launch_height']
          focal_length = row["focal_length"]

          yolo_keypoints_results = key_point_model(image)
          results_df = yolo_keypoints_results.pandas().xyxy[0]
          centers,mask_labels,mark_idx,centers_dict= get_centers(results_df)
          if not mark_idx>5:
               continue
          
          curve,img_width =  get_pixel_length(image,centers[:mark_idx])
        
          try :
               total_arc_length_pixels = curve.length
               arc_length_pixels = curve.specialize(0.2,0.7).length
          except:
               total_arc_length_pixels = 0
               arc_length_pixels = 0
               bai = 0
               length_meters = 0
               
          pixels = 0  
          if arc_length_pixels:
            mask,polygon = get_mask_yolov8(yolo_seg_model,image_path)
               
            length_meters = (altitude/focal_length)*(sensor_width/img_width)*total_arc_length_pixels

            # mask = get_roi(sam_mask_predictor,image,centers,mask_labels)
            total_whale_area = cv2.countNonZero(mask)
            pixels,percent_roi = remove_mask_quads(curve,image,mask,start,end)
            save_img(row["image"],image,mask,centers,save_folder)

          df.loc[idx,"model_length_meters"] = length_meters
          df.loc[idx,"model_length_pixel"] = total_arc_length_pixels
          df.loc[idx,'model_50_length_pixel'] = arc_length_pixels
          df.loc[idx,"pixels_roi"] = pixels
          df.loc[idx,"centers"] = [centers_dict]
          df.loc[idx,'polygon_full'] = [polygon]
          df.loc[idx,'total_whale_area']=total_whale_area
          df.loc[idx,'altitude'] = altitude  
          
     return df

def get_df(folder_path,focal_length,sensor_width,launch_height):
     df= pd.DataFrame()
     df['image'] = glob(folder_path+'/*.png')
     df["model_length_meters"] = 0
     df["model_length_pixel"] = 0
     df["pixels_roi"] = 0
     df["bai"]=0
     df['centers'] = None
     df['polygon_full'] = None
     df['total_whale_area']=None
     df['sensor_width'] = sensor_width
     df['launch_height'] = launch_height
     df['focal_length'] =focal_length
     df['altitude'] = 50
     return df

def load_models(seg_model_wt_paths = 'yolov8l-seg-ko.pt',CHECKPOINT_PATH = 'sam_vit_h_4b8939.pth'):
    # segement anything model
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    MODEL_TYPE = "vit_h"
    
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    sam.to(device=DEVICE)
    sam_mask_predictor = SamPredictor(sam)

    # yolov8 segment model
    yolo_seg_model = YOLO(seg_model_wt_paths)
    
    # keypoint model
    keypoint_yolo = torch.hub.load('ultralytics/yolov5', 'custom', path='best_7pts.pt')

    return yolo_seg_model, keypoint_yolo, sam_mask_predictor

def your_streamlit_app(lightning_app_state):
    st.title("XtraX")
    st.text("To reselect the folders, close the app and restart it")
    st.text('Give the page a few seconds to load')
    
    frames_folders_path = lightning_app_state.frames_folders_path
    results_folder_path =  lightning_app_state.results_folder_path

    yolo_seg_model, keypoint_yolo, sam_mask_predictor = load_models()
    # name of the csv where the files are saved
    name = 'results.csv'
    st.subheader("Folder to analyse")
    st.text(frames_folders_path)

    st.subheader("Folder where you want to save the results")
    st.text(results_folder_path)
    sensor_width =st.text_input('sensor_width','17.3')
    focal_length = st.text_input('focal_length','25')
    launch_height = st.text_input('launch_height','1.7')
    start  = st.text_input("start",'0.2')
    end = st.text_input('end','0.7')

    sensor_width,focal_length,launch_height ,start,end= float(sensor_width),float(focal_length),float(launch_height),float(start),float(end)

    if st.button('start'):
         df = get_df(frames_folders_path,focal_length,sensor_width,launch_height)
         df = analyse_csv(df,frames_folders_path,sensor_width,keypoint_yolo,yolo_seg_model,sam_mask_predictor,results_folder_path,start,end)
         df.to_csv(os.path.join(results_folder_path,name),index = False)

         st.success('folder analysed')
        
class SourceWork(app.LightningWork):
    def __init__(self):
        super().__init__()
        self.folder = ''
        
    def run(self):
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        messagebox.showinfo("Information", "Please select a folder to evaluate.")
        self.folder = filedialog.askdirectory()
        print(self.folder)
        root.destroy()

class DestinationWork(app.LightningWork):
    def __init__(self):
        super().__init__()
        self.folder = ''
        
    def run(self):
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        messagebox.showinfo("Information", "Please select a folder to save your evaluation.")
        self.folder = filedialog.askdirectory()
        print(self.folder)
        root.destroy()

class LitStreamlit(app.LightningFlow):

    def __init__(self):
        super().__init__()
        self.frames_folders_path = 'video folder'
        self.results_folder_path = 'save folder'

    def run(self, path1: app.storage.payload, path2: app.storage.payload):
        self.frames_folders_path = path1
        self.results_folder_path = path2

    def configure_layout(self):
        return frontend.StreamlitFrontend(render_fn=your_streamlit_app)

class LitApp(app.LightningFlow):
    def __init__(self):
        super().__init__()
        self.lit_streamlit = LitStreamlit()
        self.src = SourceWork()
        self.dst = DestinationWork()

    def run(self):
        self.src.run()
        self.dst.run()
        self.lit_streamlit.run(self.src.folder,self.dst.folder)

    def configure_layout(self):
        
        tab2 = {"name": "folders_selector","content": self.src}
        tab1 = {"name": "XtraX", "content": self.lit_streamlit}

        return [tab2,tab1]

app = app.LightningApp(LitApp())