import streamlit as st 
import numpy as np
import cv2
import onnxruntime as rt
import os
from download import download_file_from_google_drive
import time
# pip3 install streamlit opencv-python-headless onnxruntime 
# streamlit run --server.enableCORS false --server.enableXsrfProtection false --server.port 8080  segmentation.py
# https://drive.google.com/u/1/uc?export=download&confirm=hOw0&id=1-frfMZLVvyreJjhw-l1bsmBqymswmiRc

labels = {'background' : 0, 'aeroplane' : 1, 'bicycle' : 2, 'bird': 3, 'boat' :4, 'bottle': 5, 'bus': 6,
 'car': 7, 'cat': 8, 'chair': 9, 'cow': 10, 'diningtable': 11, 'dog': 12, 'horse': 13, 'motorbike': 14,
 'person': 15, 'pottedplant': 16, 'sheep': 17, 'sofa': 18, 'train': 19, 'tvmonitor': 20 }

class SegmentationInference:

    def __init__(self,path2model):
        self.sess = rt.InferenceSession(path2model)
        self.input_name = self.sess.get_inputs()[0].name
        self.input_shape = self.sess.get_inputs()[0].shape
        self.h = self.input_shape[2]
        self.w = self.input_shape[3]
        self.input_type = self.sess.get_inputs()[0].type
        self.output_name = self.sess.get_outputs()[0].name
        self.output_shape = self.sess.get_outputs()[0].shape
        self.output_type = self.sess.get_outputs()[0].type 

    def print_model_info(self):
        # Input informations
        print("input name", self.input_name)
        print("input shape", self.input_shape)
        print("input type", self.input_type)
        # Outpout informations
        print("output name", self.output_name)
        print("output shape", self.output_shape)
        print("output type", self.output_type)

    def predict(self, x):
        x = cv2.resize(x, (self.w,self.h))
        x = x.astype(np.float32)/255.0
        mean = np.array([0.485, 0.456, 0.406],dtype=np.float32)
        std = np.array([0.485, 0.456, 0.406],dtype=np.float32)
        x = x - mean
        x = x / std

        x = np.transpose(x, (2, 0, 1) ) # BCHW
        x = x[np.newaxis,:,:,:]
        x = x.astype(np.float32)
        res = self.sess.run([self.output_name], {self.input_name: x})[0]
        res = np.argmax(res, axis=1)[0]
        return res

    def replace_background(self,frame,background,idx):
        # Resize
        h = frame.shape[0]
        w = frame.shape[1]
        background = cv2.resize(background, (w,h))
        
        # Predict
        output = self.predict(frame)
        #print(output.shape,frame.shape,background.shape)
        
        # Select person
        mask = np.zeros(output.shape, dtype=np.uint8)
        mask[output == idx] = 255  # mask 0-1
        
        if idx==0:
            mask = cv2.bitwise_not(mask)

        mask = np.repeat(mask[:, :, np.newaxis], 3, axis = 2)
        
        if len(mask.flatten()==255)>20:
            kernel = np.ones((3,3), np.uint8)
            mask = cv2.erode(mask, kernel) 
        
        mask = cv2.resize(mask, (w,h))
         #display(to_pil_image(mask_t))

        # Replace background
        background[mask==255] = frame[mask==255]
        #display(to_pil_image(background))

        # Select only person
        select = cv2.bitwise_and(frame, mask)

        return background, select

@st.cache(suppress_st_warning=True)
def get_model():
    path2model = os.path.join("models","deeplabv3_resnet50.onnx")
    if not os.path.exists(path2model):
        file_id = '1OTQSLxy4Yn-ZTrB7EIFQlVWDKcQsEbZB'
        destination = 'models/deeplabv3_resnet50.onnx'
        download_file_from_google_drive(file_id, destination)
    inference = SegmentationInference(path2model)
    return inference
inference = get_model()

st.title("DeepLabV3-Resnet Segmentation")

backgrounds_path = os.path.join("images","backgrounds")
names = os.listdir(backgrounds_path)
option = st.selectbox("Select Background",names)

filename = os.path.join(backgrounds_path,option)
background = cv2.imread(filename,1)
background = cv2.cvtColor(background , cv2.COLOR_BGR2RGB)    
st.image(background, use_column_width=True)

st.set_option('deprecation.showfileUploaderEncoding', False)
uploaded_file = st.file_uploader("Reference Image", type=["png","jpeg","jpg","bmp"])
if uploaded_file is not None:
    frame = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8),cv2.IMREAD_COLOR)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.image(frame, use_column_width=True)

label = st.selectbox("Select Classes",list(labels.keys()))
btn = st.button("Combine")
if btn:
    background, select  = inference.replace_background(frame,background,idx=labels[label])
    st.image(background,use_column_width=True)
    #st.image(select,use_column_width=True)
        