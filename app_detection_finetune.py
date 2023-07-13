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
    
    def read_vid_and_save_in_folder(self,vid_path,start, end ,thresh,frame_rate, save_folder_path,display):

        vid_ca = cv2.VideoCapture(vid_path)
        extention = self.get_name_extention(vid_path)

        save_folder =save_folder_path + '/' + extention
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)

        count = 0
        b = st.button("break")
        factor = 1000/int(frame_rate)
        while vid_ca.isOpened():
            count+=1
            vid_ca.set(cv2.CAP_PROP_POS_MSEC, int(factor*count)+ 1000*int(start))
            ret, frame = vid_ca.read()
            minutes , seconds = count//60, count%60
            name = str(minutes)+'_'+str(seconds)+'_'+ extention +'.png'

            if ret :
                if display:
                    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                yolo_results = self.model(frame,size = 640)
                if len(yolo_results.xyxy[0]):
                    x_min,y_min,x_max,y_max,confi,cla = yolo_results.xyxy[0][0].numpy()
                    if confi>thresh:
                        cv2.imwrite(save_folder+'/'+name,frame)
                
            else : break

            if b:
                break
            if count==(int(end)-int(start))*int(frame_rate):
                break
    
    def render(self):
        
        st.title("Extract Frames and Save important frames")
        st.subheader("Select video")
        vid_path = st.text_input('Enter video path', '/Users/sagar/Desktop/AI_cap/sturdy-eureka/data/animated-engine/vid/vid/220904/220904_I2F_S5_U5_DJI0001.MOV')


        st.subheader("Enter location of folder where you want to save frames with whales")
        save_folder_path = st.text_input('Enter folder path', '/Users/sagar/Desktop/AI_cap/deployment/Lightning_Apps/save_detections')

        frame_rate = st.text_input('default reads one frame per second',1)
        thresh  = st.text_input('Set Detection model threshold',0.8)

        start = st.text_input("start time format : MM-SS")
        end   = st.text_input("end time format : MM-SS")
        display = st.checkbox("display frames")
        

        if st.button('start'):
            start  = start.split('-')
            start = int(start[0])*60 + int(start[1])
            end  = end.split('-')
            end = int(end[0])*60 + int(end[1])
            st.text(str(start)+ '-' + str(end))
            self.read_vid_and_save_in_folder(vid_path,start, end ,float(thresh),frame_rate, save_folder_path,display)


app = L.LightningApp(StreamlitApp())