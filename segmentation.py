import streamlit as st 
import numpy as np
import cv2
import onnxruntime as rt
import os
from download import download_file_from_google_drive

# pip3 install streamlit opencv-python-headless onnxruntime 
# streamlit run --server.enableCORS false --server.enableXsrfProtection false --server.port 8080  segmentation.py
# https://drive.google.com/u/1/uc?export=download&confirm=hOw0&id=1-frfMZLVvyreJjhw-l1bsmBqymswmiRc

path2model = os.path.join("models","deeplabv3_resnet101.onnx")
if not os.path.exists(path2model):
    file_id = '1-frfMZLVvyreJjhw-l1bsmBqymswmiRc'
    destination = 'models/deeplabv3_resnet101.onnx'
    download_file_from_google_drive(file_id, destination)

pascal_voc_labes = [
    "Person",
    "Car",
    "Bicycle",
    "Bus",
    "Motorbike",
    "Train",
    "Aeroplane",
    "Chair",
    "Bottle",
    "Dining Table",
    "Potted Plant",
    "TV/Monitor",
    "Sofa",
    "Bird",
    "Cat",
    "Cow",
    "Dog",
    "Horse",
    "Sheep"
]

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
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis = 2)
        kernel = np.ones((25,25), np.uint8)
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
  inference = SegmentationInference(path2model)
  return inference
inference = get_model()

st.title("DeepLabV3-Resnet101 Segmentation")

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

btn = st.button("Combine")
if btn:
    background, select  = inference.replace_background(frame,background,idx=15)
    st.image(background,use_column_width=True)
    #st.image(select,use_column_width=True)
        