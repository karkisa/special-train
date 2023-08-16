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
# from streamlit_drawable_canvas import st_canvas
import pandas as pd
from scipy.special import comb
import bezier

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

        keypoint_yolo = torch.hub.load('ultralytics/yolov5', 'custom', path='best_7pts.pt')
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
    
    def B(self,i, N, t):
        val = comb(N,i) * t**i * (1.-t)**(N-i)
        return val
        
    def P(self,t, X):
        '''
        xx = P(t, X)
        
        Evaluates a Bezier curve for the points in X.
        
        Inputs:
        X is a list (or array) or 2D coords
        t is a number (or list of numbers) in [0,1] where you want to
            evaluate the Bezier curve
        
        Output:
        xx is the set of 2D points along the Bezier curve
        '''
        X = np.array(X)
        N,d = np.shape(X)   # Number of points, Dimension of points
        N = N - 1
        xx = np.zeros((len(t), d))
        
        for i in range(N+1):
            xx += np.outer(self.B(i, N, t), X[i])
        return xx
    
    def get_bezire_curve(self,img,centers,alt):
        n_points = st.text_input("number of points for bezier curve",200)
        distance = np.linspace(0, 1, int(n_points))
        curve_points = self.P(distance, centers)
        nodes1 = np.asfortranarray([
            curve_points[:,0], curve_points[:,1]
            ])
        st.text(f"[height,width,channels] of image : {img.shape}")
        curve1 = bezier.Curve(nodes1, degree=len(curve_points)-1)
        
        for x,y in curve_points:
            cv2.circle(img, tuple((int(x),int(y))), 1, (255,0,0),10)
        st.image(img)

        return curve_points,curve1
    
    def get_centers(self,df):
        conf = [-1]*6
        fins = []
        centers = [[0,0]]*6
        mask_labels = []
        if df[df["class"]!=6]["class"].nunique()<6:
            st.text("Not all 6 key points detected. Try generating more frames using finetnuing app")
            return [],None
        for idx , row in df.iterrows():
            key_pt = [int((row['xmax'] + row['xmin'])/2), int((row['ymax'] + row['ymin'])/2)]
            
            if row['class']==6:
                fins.append(key_pt)

            else:
                if centers[row['class']]==[0,0] or row["confidence"]>conf[row['class']] :
                    centers[row['class']]=key_pt
                    mask_labels.append(1)  # 1 vor visibilty

        for pts in fins:
            centers.append(pts)
            mask_labels.append(0)   # point not included in mask

        return np.array(centers),np.array(mask_labels)

    def detection_op(self,yolo_results, yolo_keypoints_results,image):

        st.write(yolo_results.pandas().xyxy[0])
        st.write(yolo_keypoints_results.pandas().xyxy[0])
        centers,mask_labels = self.get_centers(yolo_keypoints_results.pandas().xyxy[0])
        x_min,y_min,x_max,y_max,confi,cla = yolo_results.xyxy[0][0].numpy()
        
        color = (255, 0, 0)
        # Line thickness of 2 px
        thickness = 2
        img_copy = np.copy(image)
        
        if  len(centers)==0:
            return None,[],None

        for point in centers:
            cv2.circle(img_copy, tuple(point), 1, (255,0,0),10)
        cv2.rectangle(img_copy,[int(x_min),int(y_min)],[int(x_max),int(y_max)],color, thickness)
        st.image(img_copy) 
        box_cordinates = np.array([x_min,y_min,x_max,y_max])
        return box_cordinates,centers,mask_labels
    
    def get_roi(self,mask_predictor,image,centers,mask_labels):
          mask_predictor.set_image(image)
          masks, scores, logits = mask_predictor.predict(
            # box=box_cords,
            point_coords=centers,
            point_labels=mask_labels,
            multimask_output=False
                )
          mask = masks[0].astype(float)
          st.image(mask)
          return  mask
    
    def analyse_mask_basic(self,image,mask):
            pixels = cv2.countNonZero(mask)
            st.text(f"pixels covered by seleted area {pixels}"
                )
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

    def analyse_curve(self,curve,image,alt):
        arc_length_pixels = curve.length
        st.text(f"length in pixel =  {arc_length_pixels}")
        sensor_width  = eval(st.text_input("Set camera sensor width in mm", 17.3))
        focal_length = eval(st.text_input("Set focal length in mm ",25 ))
        launch_hieght = eval(st.text_input("Set launch altitude in meters", 0))
        altitude = eval(st.text_input("Set altitude in meters", alt+launch_hieght))
        img_width = image.shape[1]
        st.text(f"found image width ot be : {img_width}")
        length_meters = (altitude/focal_length)*(sensor_width/img_width)*arc_length_pixels
        st.text(f"The Length in meters is {length_meters}")

        return
    
    def get_prompts(self,curve_points,mask):
        ln = len(curve_points)
        start = eval(st.text_input("set start",0))
        end = eval(st.text_input("set end",100))
        start = (start*ln)//100
        end  = (end*ln)//100
        pt_masks  = torch.ones((ln))
        pt_masks[:start]=0
        pt_masks[end:] = 0
        return pt_masks

    def render_straight_cuts(self,mask,curve):
        start = eval(st.text_input("set start",1.25))
        end = eval(st.text_input("set end",.8))

        x1,y1 = curve.evaluate(start)
        x2,y2 = curve.evaluate(end)
        if abs(x2-x1)>abs(y2-y1):

            mask[:,:int((min(x2,x1)))]=0
            mask[:,int(max(x2,x1)):] = 0

        else:
            mask[:int((min(y2,y1))),:]=0
            mask[int(max(y2,y1)):,:] = 0
            
        st.image(mask)

    def get_dir(self,p1,p2):
        if p1<p2:
            return 1
        return -1
    
    def remove_mask_quads(self,curve,image,mask):
        start = eval(st.text_input("set start",0.25))
        end = eval(st.text_input("set end",.8))

        x1,y1 = curve.evaluate(start)
        x2,y2 =  curve.evaluate(end)
        
        if abs(x2-x1)>abs(y2-y1):
            direction = self.get_dir(x1,x2)
        else:
            direction = self.get_dir(y1,y2)

        pt_set1 = self.get_quadpts(start,0.0,direction,curve,image,mask)
        pt_set2 = self.get_quadpts(end,1.0,-direction,curve,image,mask)
        return
    

    def get_quadpts( self,percent,end_pt_ext,direction,curve,image,mask):
        
        box_d = 0.15*curve.length
        extra_length = direction*0.2*curve.length

        sn,sd =  curve.evaluate_hodograph(percent)
        slope =  -(sd/sn)

        x1,y1 = curve.evaluate(percent)

        pt1 = self.get_point_k_dist(slope,[x1,y1],box_d)
        pt2 = self.get_point_k_dist(slope,[x1,y1],-box_d)
       

        x00,y00 = curve.evaluate(end_pt_ext)
        sn,sd =  curve.evaluate_hodograph(0)
        slope2 = -(sd/sn)
        ext_pt = self.get_point_k_dist((sn/sd),[x00,y00],extra_length)
        cv2.circle(image,ext_pt,1, (255,0,0),10)

        pt3 = self.get_point_k_dist(slope,ext_pt,-box_d)
        pt4 = self.get_point_k_dist(slope,ext_pt,box_d)

        cv2.circle(image,pt1,1, (255,0,0),10)
        cv2.circle(image,pt2,1, (255,255,0),10)
        cv2.circle(image,pt3,1, (0,0,255),10)
        cv2.circle(image,pt4,1, (255,0,255),10)

        st.image(image)
        points = np.array([pt1,pt2,pt3,pt4])
        
        cv2.fillPoly(mask,pts=np.int32([points]), color=0)
        st.image(mask)
        
        return [pt1,pt2,pt3,pt4]

    def get_point_k_dist(self,slope,pt,d):
        x1,y1 = pt
        x2  = x1+d*np.sin(np.arctan(slope))
        y2 = y1 +d*np.cos(np.arctan(slope))

        return [int(x2),int(y2)]

    def render(self):
        st.title("Morphology Assistant")
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file:
            alt = uploaded_file.name.split("_")[3:5]
            try :
                alt = float(alt[0])+float(alt[1])/(10*len(alt[1]))

            except:
                alt = 50

        if uploaded_file is not None:
          
          image = st.image(uploaded_file,use_column_width=True)
          image = self.read_img(uploaded_file)

          mask_predictor , yolo_model , keypoint_yolo = self.model
          yolo_results  =yolo_model(image,size = 640)
          yolo_keypoints_results = keypoint_yolo(image, size= 640)
 
          box_cords,centers,mask_labels = self.detection_op(yolo_results,yolo_keypoints_results,image)
          if len(centers):
            curve_points, curve = self.get_bezire_curve(image.copy(),centers[:6],alt)  # centers on ly first 5 points because rest of the points are on fin. The first 6 are on central body.
            self.analyse_curve(curve,image,alt)
            mask = self.get_roi(mask_predictor,image,centers,mask_labels)
            self.remove_mask_quads(curve,image,mask)
          
        
app = L.LightningApp(StreamlitApp())
