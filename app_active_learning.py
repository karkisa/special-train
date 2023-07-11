import lightning as L
import streamlit as st
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from streamlit_drawable_canvas import st_canvas
import pandas as pd
from PIL import Image
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
        
        return 
    

    def render(self):
        st.title("Morphology Assistant")
        uploaded_file = st.file_uploader("Choose a file")

        if uploaded_file is not None:

          
            # Specify canvas parameters in application
            drawing_mode = st.sidebar.selectbox(
            "Drawing tool:", ("point", "freedraw", "line", "rect", "circle", "transform")
            )

            stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
            if drawing_mode == 'point':
                  point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
            stroke_color = st.sidebar.color_picker("Stroke color hex: ")
            bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
            # bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])

            realtime_update = st.sidebar.checkbox("Update in realtime", True)

            

            # Create a canvas component
            canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            background_image=Image.open(uploaded_file) if uploaded_file else None,
            update_streamlit=realtime_update,
            height=400,
            width=700,
            drawing_mode=drawing_mode,
            point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
            key="canvas",
            )

            # Do something interesting with the image data and paths
            # if canvas_result.image_data is not None:
                  # st.image(canvas_result.image_data)
            if canvas_result.json_data is not None:
                  objects = pd.json_normalize(canvas_result.json_data["objects"]) # need to convert obj to str because PyArrow
            for col in objects.select_dtypes(include=['object']).columns:
                  objects[col] = objects[col].astype("str")
            st.dataframe(objects)

            st.image(uploaded_file)
          
        
app = L.LightningApp(StreamlitApp())
