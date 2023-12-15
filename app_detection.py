import lightning as L
import streamlit as st
import pandas as pd
import cv2, torch, warnings,os
import lightning_app as app       
from glob import glob
import wx
import easygui
from PyQt5.QtWidgets import QApplication, QFileDialog
import tkinter as tk
from tkinter import filedialog
from lightning.app import LightningFlow


warnings.filterwarnings('ignore')

class StreamlitApp(app.components.ServeStreamlit):

    def build_model(self):
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
        
        return model
    
    def get_df(self,path):

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
    
    def get_name_extention(self,path):
        path=path.split('/')
        s=path[-1]
        s = s.split('.')
        return s[0]
    
    def read_vid_and_save_in_folder(self,vid_path,df,parent_folder = '/nfs/hpc/share/karkisa/AI cap/sturdy-eureka/yolo_training_data/test_data/210914'):
        vid_ca = cv2.VideoCapture(vid_path)
        extention = self.get_name_extention(vid_path)
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
            name =extention +"_000_"+ str(minutes)+'_'+str(seconds)+'_000_'+alt +'.png'

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
        st.title("DeteX")
        st.subheader("Enter location of the folder that contains videos")

        vid_folders_path = st.text_input("select source folder")
        st.subheader("Enter location of folder where you want to save frames with whales")
        save_folder_path =  st.text_input("select destination folder")
        st.echo(save_folder_path)

        if st.button('start'):
            list_folders = glob(os.path.join(vid_folders_path,"**/*.MOV"),recursive=True)
            
            for vid_path in list_folders:
                df = self.get_df(vid_path)
                if len(df):
                    self.read_vid_and_save_in_folder(vid_path,df,save_folder_path)

                st.success(f'{vid_path.split("/")[-1]} analysed')

app = app.LightningApp(StreamlitApp())
