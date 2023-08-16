import pandas as pd
import torch
import cv2,os
import numpy as np
from PIL import Image
import bezier
from scipy.special import comb

def get_centers(df):
        conf = [-1]*6
        fins = []
        centers = [[0,0]]*6

        if df[df["class"]!=6]["class"].nunique()<6:
            # st.text("Not all 6 key points detected. Try generating more frames using finetnuing app")
            return []
        
        for idx , row in df.iterrows():
            key_pt = [int((row['xmax'] + row['xmin'])/2), int((row['ymax'] + row['ymin'])/2)]
            
            if row['class']==6:
                fins.append(key_pt)

            else:
                if centers[row['class']]==[0,0] or row["confidence"]>conf[row['class']] :
                    centers[row['class']]=key_pt

        return np.array(centers)

def get_bezire_curve(centers):
        n_points = 100
        distance = np.linspace(0, 1, int(n_points))
        curve_points = P(distance, centers)
        nodes1 = np.asfortranarray([
            curve_points[:,0], curve_points[:,1]
            ])
        curve1 = bezier.Curve(nodes1, degree=len(curve_points)-1)

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

def get_pixel_length(image_path,model):
    
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    yolo_keypoints_results = model(image, size= 640)
    results_df = yolo_keypoints_results.pandas().xyxy[0]
    centers = get_centers(results_df)
    if len(centers)==0:
         return 0,image.shape[1]
    curve = get_bezire_curve(centers)

    return curve.length,image.shape[1]

def analyse_csv(df,base_folder,sensor_width,model):
     df["model_length_meters"] = 0
     df["model_length_pixel"] = 0
     df["expected_length_meters"] = 0
     for idx,row in df.iterrows():
          image_path = os.path.join(base_folder,row["Image"])
          altitude = row["Altitude"]
          focal_length = row["Focal_Length"]
          arc_length_pixels,img_width =  get_pixel_length(image_path,model)
          length_meters = (altitude/focal_length)*(sensor_width/img_width)*arc_length_pixels
          df.loc[idx,"model_length_meters"] = length_meters
          df.loc[idx,"model_length_pixel"] = arc_length_pixels
          df.loc[idx,"expected_length_meters"] = (altitude/focal_length)*(sensor_width/img_width)*row["TL"]

     return df

if __name__=="__main__":
    csv_path = '/Users/sagar/Desktop/AI_cap/deployment/Lightning_Apps/save_detections/Kodiak_mx_whales_cx.xlsx'
    base_folder = '/Users/sagar/Desktop/AI_cap/deployment/Lightning_Apps/save_detections/Kos_measurements'
    sensor_width = 17.3
    keypoint_yolo = torch.hub.load('ultralytics/yolov5', 'custom', path='best_7pts.pt')
    df = pd.read_excel(csv_path)
    df = analyse_csv(df,base_folder,sensor_width,keypoint_yolo)
    df.to_csv("save_detections/name.csv")