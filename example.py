import streamlit as st 
import pandas as pd
import numpy as np
import cv2
import onnxruntime as rt
from PIL import Image
import json

def softmax(vector):
	e = np.exp(vector)
	return e / e.sum()

def apply(im,mi):
  im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
  iW = im.shape[1]
  iH = im.shape[0]
  x = cv2.resize(im, (224,224))
  x = x.astype(np.float32)/255.0
  x = np.transpose(x, (2, 0, 1) ) # BCHW
  x = x[np.newaxis,:,:,:]
  res = mi.predict(x)
  return res

class modelInference:

  def __init__(self,path2model):
    self.sess = rt.InferenceSession(path2model)
    self.input_name = self.sess.get_inputs()[0].name
    self.input_shape = self.sess.get_inputs()[0].shape
    self.input_type = self.sess.get_inputs()[0].type
    self.output_name = self.sess.get_outputs()[0].name
    self.output_shape = self.sess.get_outputs()[0].shape
    self.output_type = self.sess.get_outputs()[0].type
    with open('models/imagenet_labels.json') as f:
        self.labels = json.load(f)
  
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
    x = x.astype(np.float32)
    res = self.sess.run([self.output_name], {self.input_name: x})[0]
    probs = softmax(res)[0]
    idx = np.argsort(-probs,axis=-1)
    idxTop = idx[0:5]
    
    s = ""
    for i in idxTop:
        p = np.round(probs[i]*100,2)
        s+=f"{self.labels [i]}: {p}%, "

    #res = self.labels[res]
    return  s

@st.cache(suppress_st_warning=True)
def get_model():
  mi = modelInference(path2model="models/classification.onnx")
  return mi
mi = get_model()


st.title("Resnet18 Classification")

# ------------
st.sidebar.title("Parametri HSV filtraggio colori")
Hmin,Hmax = st.sidebar.slider("Hmin - Hmax", 0, 255, (0, 255), 1)
Smin,Smax = st.sidebar.slider("Smin - Smax", 0, 255, (0, 255), 1)
Vmin,Vmax = st.sidebar.slider("Vmin - Vmax", 0, 255, (0, 255), 1)

st.set_option('deprecation.showfileUploaderEncoding', False)
uploaded_file = st.file_uploader("Upload Image", type=["png","jpeg","jpg","bmp"])
if uploaded_file is not None:
    im = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8),cv2.IMREAD_COLOR)

    # ----
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    # Threshold of blue in HSV space
    lower_blue = np.array([Hmin,Smin,Vmin])
    upper_blue = np.array([Hmax,Smax,Vmax])
    # preparing the mask to overlay
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    result = cv2.bitwise_and(im, im, mask = mask)
    # ---
  
    st.image(result, use_column_width=True)
#------------

btn = st.button("Predict")
if btn:
    # Classificazione
    res = apply(im,mi)
    st.write(res)
  