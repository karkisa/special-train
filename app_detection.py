# app.py
# !pip install streamlit omegaconf scipy
# !pip install torch
import lightning as L
import streamlit as st
import cv2, torch, warnings,os
warnings.filterwarnings('ignore')

class StreamlitApp(L.app.components.ServeStreamlit):
    def build_model(self):
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
        return model
    
    def get_name_extention(self,path):
        path=path.split('/')
        s=path[-1]
        s = s.split('.')
        return s[0]
    
    def read_vid_and_save_in_folder(self,vid_path,parent_folder = '/nfs/hpc/share/karkisa/AI cap/sturdy-eureka/yolo_training_data/test_data/210914'):
        vid_ca = cv2.VideoCapture(vid_path)
        extention = self.get_name_extention(vid_path)
        save_folder =parent_folder + '/' + extention
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        count = 0
        time_skips = 3000             # fps is 29.9 .i.e almost 30
        while vid_ca.isOpened():
            count+=1
            t_msec = 1000*(count)
            vid_ca.set(cv2.CAP_PROP_POS_MSEC, t_msec)
            ret, frame = vid_ca.read()
            name = str(count)+'_'+ extention +'.png'
            # st.progress(count)

            if ret :
                yolo_results = self.model(frame,size = 640)
                if len(yolo_results.xyxy[0]):
                    x_min,y_min,x_max,y_max,confi,cla = yolo_results.xyxy[0][0].numpy()
                    if confi>0.8:
                        cv2.imwrite(save_folder+'/'+name,frame)
                
            else : break
    
    def render(self):
        st.title("Convert the vids to images")
        st.subheader("Get the video folder path")
        vid_folder_path = st.text_input('Enter folder path', '/Users/sagar/Desktop/AI_cap/sturdy-eureka/data/animated-engine/vid/220901')
        st.subheader("Get the video folder path")
        save_folder_path = st.text_input('Enter folder path', '/Users/sagar/Desktop/AI_cap/deployment/Lightning_Apps/save_detections')
        list_p = os.listdir(vid_folder_path)
        list_p = [os.path.join(vid_folder_path,p) for p in list_p]
        
        for vid_path in list_p:
            self.read_vid_and_save_in_folder(vid_path,save_folder_path)

        
app = L.LightningApp(StreamlitApp())
