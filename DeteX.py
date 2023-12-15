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

def get_df(path):
    os.system(f"exiftool -ee {path} > output.txt")
    df = pd.DataFrame(columns=["Sec","H"])
    with open('output.txt') as file:
        i = -1
        
        for line in file:
            if "Sample Time" in line:
                i+=1
                if 's' in line:
                    s = float(line.split(':')[1].split('s')[0])
                else:
                    s  = line.split(':')
                    s = int(s[-2])*60 + int(s[-1])
                df.loc[i] = [s,0]
                
            if "GPS Altitude" in line  and "Ref" not in  line:
                if df.loc[i][1]==0: # there is an extra "GPS Altitude" in srt file towards the end
                    df.loc[i][1] = float(line.split(':')[1].split('m')[0])

    os.system("rm output.txt")
    return df
   
def get_name_extention(path):
    path=path.split('/')
    s=path[-1]
    s = s.split('.')
    return s[0]
    
def read_vid_and_save_in_folder(vid_path,df,parent_folder,model):
    vid_ca = cv2.VideoCapture(vid_path)
    extention = get_name_extention(vid_path)
    st.text(f"Going through {extention}.MOV")
    save_folder =parent_folder + '/' + extention
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    ln = len(df)
    count = 0
    while vid_ca.isOpened():
        count+=1
        t_msec = 1000*(count)
        vid_ca.set(cv2.CAP_PROP_POS_MSEC, t_msec)
        ret, frame = vid_ca.read()
        minutes , seconds = count//60, count%60
        
        if count<ln:

            alt = ("_").join(str(df["H"][count]).split("."))
        name =extention +"_000_00_"+ str(minutes)+'_'+str(seconds)+'_000_'+alt +'.png'

        if ret :
            yolo_results = model(frame,size = 640)
            if len(yolo_results.xyxy[0]):
                x_min,y_min,x_max,y_max,confi,cla = yolo_results.xyxy[0][0].numpy()
                if confi>0.8:
                    cv2.imwrite(save_folder+'/'+name,frame)
            
        else :
                st.success(f'{extention}.MOV  analysed')
                break
  
def your_streamlit_app(lightning_app_state):
    st.title("DeteX")
    st.subheader("Reload page to reflect your selection")
    
    vid_folders_path = lightning_app_state.vid_folder
    save_folder_path =  lightning_app_state.save_folder
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

    st.subheader("Folder to analyse")
    st.text(vid_folders_path)

    st.subheader("Folder where you want to save the frames")
    st.text(save_folder_path)

    if st.button('start'):
        list_folders = glob(os.path.join(vid_folders_path,"**/*.MOV"),recursive=True)
        
        for vid_path in list_folders:
            df = get_df(vid_path)
            if len(df):
                read_vid_and_save_in_folder(vid_path,df,save_folder_path,model)

            st.success(f'{vid_path.split("/")[-1]} analysed')

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

class DestinationWork(app.LightningWork):
    def __init__(self):
        super().__init__()
        self.folder = ''
        
    def run(self):
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        messagebox.showinfo("Information", "Please select a folder to save the results. The application will create saperate folders inside the selected folder for every video where it will save the frames.")
        self.folder = filedialog.askdirectory()
        print(self.folder)

class LitStreamlit(app.LightningFlow):

    def __init__(self):
        super().__init__()
        self.vid_folder = 'video folder'
        self.save_folder = 'save folder'

    def run(self, path1: app.storage.payload, path2: app.storage.payload):
        self.vid_folder = path1
        self.save_folder = path2

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
        tab1 = {"name": "DeteX", "content": self.lit_streamlit}
        tab2 = {"name": "video_folder_selector","content": self.src}
        return [tab2,tab1]

app = app.LightningApp(LitApp())