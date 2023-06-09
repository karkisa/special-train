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
        st.text(f"Going through {extention}.MOV")
        save_folder =parent_folder + '/' + extention
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        count = 0
        while vid_ca.isOpened():
            count+=1
            t_msec = 1000*(count)
            vid_ca.set(cv2.CAP_PROP_POS_MSEC, t_msec)
            ret, frame = vid_ca.read()
            minutes , seconds = count//60, count%60
            name = str(minutes)+'_'+str(seconds)+'_'+ extention +'.png'

            if ret :
                yolo_results = self.model(frame,size = 640)
                if len(yolo_results.xyxy[0]):
                    x_min,y_min,x_max,y_max,confi,cla = yolo_results.xyxy[0][0].numpy()
                    if confi>0.8:
                        cv2.imwrite(save_folder+'/'+name,frame)
                
            else :
                 st.success(f'{extention}.MOV  analysed')
                 break
    
    def render(self):
        st.title("Extract Frames and Save important frames")
        st.subheader("Enter location of the folder that contains videos")
        vid_folders_path = st.text_input("select sorce folder")
        st.subheader("Enter location of folder where you want to save frames with whales")
        save_folder_path =  st.text_input("select destination folder")
        st.echo(save_folder_path)
        if st.button('start'):
            list_folders = os.listdir(vid_folders_path)
            list_folders =[os.path.join(vid_folders_path,p) for p in list_folders]
            
            for vid_folder_path in list_folders:
                st.text(f'Analysing {vid_folder_path.split("/")[-1]} folder')
                list_p = os.listdir(vid_folder_path)
                list_p = [os.path.join(vid_folder_path,p) for p in list_p]
                print(list_p)
                for vid_path in list_p:
                    self.read_vid_and_save_in_folder(vid_path,save_folder_path)
                st.success(f'{vid_folder_path.split("/")[-1]} folder analysed')

app = L.LightningApp(StreamlitApp())
