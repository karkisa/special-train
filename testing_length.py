import pandas as pd
import torch
import cv2,os
import numpy as np
from PIL import Image
import bezier
from scipy.special import comb
from segment_anything import  SamAutomaticMaskGenerator,sam_model_registry, SamPredictor
import collections,math
import pdb
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

    return np.array(center_pts),np.array(mask_labels),mark_idx

def get_bezire_curve(centers):
        # n_points = 100
        # distance = np.linspace(0, 1, int(n_points))
        # curve_points = P(distance, centers)
        # nodes1 = np.asfortranarray([
        #     curve_points[:,0], curve_points[:,1]
        #     ])
        
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

def get_roi(mask_predictor,image,centers,mask_labels):
          mask_predictor.set_image(image)
          masks, scores, logits = mask_predictor.predict(
            # box=box_cords,
            point_coords=centers,
            point_labels=mask_labels,
            multimask_output=False
                )
          mask = masks[0].astype(float)
        #   st.image(mask)
          return  mask

def get_dir(p1,p2):
    if p1<p2:
        return 1
    return -1

def remove_mask_quads(curve,image,mask):
    start = 0.20 
    end = 0.7

    x1,y1 = curve.evaluate(start)
    x2,y2 =  curve.evaluate(end)

    print(cv2.countNonZero(mask))
    pt_set1 = get_quadpts(start,0.0,curve,image,mask)
    print(cv2.countNonZero(mask))
    pt_set2 = get_quadpts(end,1.0,curve,image,mask)
    print(cv2.countNonZero(mask))
    pixels = cv2.countNonZero(mask)

    return pixels, (end-start)

def save_img(name,image,mask,centers,base_save_folder):
     
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

def analyse_csv(df,base_folder,sensor_width,model,mask_predictor,save_folder):
     df["model_length_meters"] = 0
     df["model_length_pixel"] = 0
     df["pixels_roi"] = 0
     df["bai"]=0
     for idx,row in df.iterrows():
          
          image_path = os.path.join(base_folder,row["image"])
          if not os.path.exists(image_path):
               continue
          image = get_image(image_path)

          altitude = row['Baro_Alt']
          focal_length = row["Focal_Length"]

          yolo_keypoints_results = model(image, size = 640)
          results_df = yolo_keypoints_results.pandas().xyxy[0]
          centers,mask_labels,mark_idx= get_centers(results_df)
          
          curve,img_width =  get_pixel_length(image,centers[:mark_idx])
          
          if curve:
               centers_2 = curve.locate(np.array([[float(centers[1][0])],[float(centers[1][1])]]))  
               print(np.array([[float(centers[1][0])],[float(centers[1][1])]]))
          else: centers_2 = np.array([0.0,0])
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
               
            length_meters = (altitude/focal_length)*(sensor_width/img_width)*total_arc_length_pixels

            mask = get_roi(mask_predictor,image,centers,mask_labels)
            pixels,percent_roi = remove_mask_quads(curve,image,mask)
            save_img(row["image"],image,mask,centers,save_folder)

          print(centers_2,total_arc_length_pixels,arc_length_pixels,pixels)

          df.loc[idx,"model_length_meters"] = length_meters
          df.loc[idx,"model_length_pixel"] = total_arc_length_pixels
          df.loc[idx,'model_50_length_pixel'] = arc_length_pixels
          df.loc[idx,"pixels_roi"] = pixels

     return df

if __name__=="__main__":
    csv_path = 'test/GRANITE_230526_to_230724/snapshots/manual_measurements.csv'
    base_folder = 'test/GRANITE_230526_to_230724/snapshots'
    save_folder = 'results/GRANITE_230526_to_230724'
    sensor_width = 17.3
    keypoint_yolo = torch.hub.load('ultralytics/yolov5', 'custom', path='best_7pts.pt')
    name = 'results.csv'
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    MODEL_TYPE = "vit_h"
    CHECKPOINT_PATH = 'sam_vit_h_4b8939.pth'

    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    sam.to(device=DEVICE)
    mask_predictor = SamPredictor(sam)

    df = pd.read_csv(csv_path)
    df = analyse_csv(df,base_folder,sensor_width,keypoint_yolo,mask_predictor,save_folder)
    df.to_csv(os.path.join(save_folder,name),index = False)