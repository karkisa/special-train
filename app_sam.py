# app.py
# !pip install streamlit omegaconf scipy
# !pip install torch
from lightning_app import LightningApp
# import lightning as L
import torch, math
# from io import BytesIO
# from functools import partial
import streamlit as st
# import warnings
import numpy as np
from PIL import Image
# from segmentation_models_pytorch import Unet
from segment_anything import  SamAutomaticMaskGenerator,sam_model_registry, SamPredictor
import torch
import cv2
# from skimage.morphology import skeletonize
# from sklearn.metrics.pairwise import euclidean_distances
# from streamlit_drawable_canvas import st_canvas
# import pandas as pd
from scipy.special import comb
import bezier, collections
# warnings.filterwarnings('ignore')
import lightning_app as  app       

class StreamlitApp(app.components.ServeStreamlit):
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
        st.text(len(center_pts))
        return np.array(center_pts),np.array(mask_labels),mark_idx

    def detection_op(self,yolo_results, yolo_keypoints_results,image):

        st.write(yolo_results.pandas().xyxy[0])
        st.write(yolo_keypoints_results.pandas().xyxy[0])
        centers,mask_labels,mark_idx = self.get_centers(yolo_keypoints_results.pandas().xyxy[0])
        x_min,y_min,x_max,y_max,confi,cla = yolo_results.xyxy[0][0].numpy()
        
        color = (255, 0, 0)
        # Line thickness of 2 px
        thickness = 2
        img_copy = np.copy(image)
        
        if len(centers)==0:
            return None,[],None,None

        for point in centers:
            cv2.circle(img_copy, tuple(point), 1, (255,0,0),10)
        cv2.rectangle(img_copy,[int(x_min),int(y_min)],[int(x_max),int(y_max)],color, thickness)
        st.image(img_copy) 
        box_cordinates = np.array([x_min,y_min,x_max,y_max])
        return box_cordinates,centers,mask_labels,mark_idx
    
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
    
    def midpoint(self,ptA, ptB):
        return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

    def analyse_mask_basic(self,image,mask):
            pixels = cv2.countNonZero(mask)
            st.text(f"pixels covered by seleted area {pixels}"
                )
            contours, _ = cv2.findContours(np.array(mask, np.uint8), cv2.RETR_TREE,
                                cv2.CHAIN_APPROX_SIMPLE)
            cnt = max(contours, key = cv2.contourArea)
            
            

            # take the first contour
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            # Line thickness of 2 px
            (tl, tr, br, bl) = box

            (tltrX, tltrY) = self.midpoint(tl,tr)
            (blbrX, blbrY) = self.midpoint(bl,br)
            (tlblX, tlblY) = self.midpoint(tl, bl)
            (trbrX, trbrY) = self.midpoint(tr, br)

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
        
        curve2 = curve.specialize(0.2,0.7)
        st.text(curve2.length)

        return length_meters , arc_length_pixels
    
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
        st.text(str(p1)+"_"+str(p2))
        if p1<p2:
            return 1
        return -1
    
    def remove_mask_quads(self,curve,image,mask):
        start = eval(st.text_input("set start",0.20))
        end = eval(st.text_input("set end",0.70))

        x1,y1 = curve.evaluate(start)
        x2,y2 =  curve.evaluate(end)
        st.text_area(str(abs(x2-x1)))
        st.text_area(str(abs(y2-y1)))
        

        pt_set1 = self.get_quadpts(start,0.0,curve,image,mask)
        pt_set2 = self.get_quadpts(end,1.0,curve,image,mask)
        mask2= mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY )[1]
        image[mask2==255] = (255,0,0)
        st.image(image)
        pixels = cv2.countNonZero(mask)
        st.text(f"pixels covered by seleted area {pixels}"
                )
        
        return pixels, (end-start)
    
    def get_quadpts( self,percent,end_pt_ext,curve,image,mask):
        direction = -1 if end_pt_ext==0 else 1
        box_d = 0.15*curve.length
        extra_length = direction*0.2*curve.length

        vx,vy =  curve.evaluate_hodograph(percent)
        x1,y1 = curve.evaluate(percent)
        # clockwise - > [1,-1]
        pt1 = self.get_point_k_dist(vy,-vx,[x1,y1],box_d)
        pt2 = self.get_point_k_dist(-vy,vx,[x1,y1],box_d)
       

        x00,y00 = curve.evaluate(end_pt_ext)
        vx,vy =  curve.evaluate_hodograph(end_pt_ext)
        
        ext_pt = self.get_point_k_dist(vx,vy,[x00,y00],extra_length)
        # cv2.circle(image,ext_pt,1, (255,0,0),10)

        pt3 = self.get_point_k_dist(-vy,vx,ext_pt,box_d)
        pt4 = self.get_point_k_dist(vy,-vx,ext_pt,box_d)

        # cv2.circle(image,[int(x1),int(y1)],1, (255,0,0),10)
        # cv2.line(image, pt1, pt2, (255,0,0), 5) 
        # cv2.line(image, pt2, pt3, (255,0,0), 5) 
        # cv2.line(image, pt3, pt4, (255,0,0), 5) 
        # cv2.line(image, pt4, pt1, (255,0,0), 5) 
        

        # cv2.circle(image,pt1,1, (255,0,0),10)
        # cv2.circle(image,pt2,1, (255,255,0),10)
        # cv2.circle(image,pt3,1, (0,0,255),10)
        # cv2.circle(image,pt4,1, (255,0,255),10)

        st.image(image)
        points = np.array([pt1,pt2,pt3,pt4])
        
        cv2.fillPoly(mask,pts=np.int32([points]), color=0)
        
        st.image(mask)
        pixels = cv2.countNonZero(mask)
        st.text(f"pixels covered by seleted area {pixels}"
                )
        
        return [pt1,pt2,pt3,pt4]

    def get_point_k_dist(self,vx,vy,pt,d):
        x1,y1 = pt
        
        magnitude = math.sqrt(vx**2 + vy**2)
        x2 = x1 + (vx/magnitude) * d
        y2 = y1 + (vy/magnitude) * d
    
        return [int(x2),int(y2)]

    def render(self):
        st.title("XtraX")
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
          yolo_results  = yolo_model(image,size = 640)
          yolo_keypoints_results = keypoint_yolo(image, size= 640)
 
          box_cords,centers,mask_labels,mark_idx = self.detection_op(yolo_results,yolo_keypoints_results,image)
          if len(centers):
            curve_points, curve = self.get_bezire_curve(image.copy(),centers[:mark_idx],alt)  # centers on ly first 5 points because rest of the points are on fin. The first 6 are on central body.
            length_meters , arc_length_pixels = self.analyse_curve(curve,image,alt)
            mask = self.get_roi(mask_predictor,image,centers,mask_labels)
            pixels,percent_roi = self.remove_mask_quads(curve,image,mask)
            st.text(f"BAI is {(pixels/((arc_length_pixels*percent_roi)**2)*100)}")

app = LightningApp(StreamlitApp())
