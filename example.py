import streamlit as st 
import pandas as pd
import numpy as np
import cv2

import onnxruntime as rt
import numpy as np
import cv2

def apply(im,mi):
  im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
  iW = im.shape[1]
  iH = im.shape[0]
  #print(w,h)
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
    res = np.argmax(res, axis=1)[0]
    #res = self.labels[res]
    return res

@st.cache(suppress_st_warning=True)
def get_model():
  mi = modelInference(path2model="models/classification.onnx")
  return mi
mi = get_model()


st.title("HSV Filtering App")

# ------------
st.sidebar.title("Parametri")
hmin = st.sidebar.slider('H min value', 0, 255, 55)
smin = st.sidebar.slider('S min value', 0, 255, 142)
vmin = st.sidebar.slider('V min value', 0, 255, 134)
hmax = st.sidebar.slider('H max value', 0, 255, 255)
smax = st.sidebar.slider('S max value', 0, 255, 255)
vmax = st.sidebar.slider('V max value', 0, 255, 255)

st.set_option('deprecation.showfileUploaderEncoding', False)
uploaded_file = st.file_uploader("Upload Image", type=["png","jpeg","jpg","bmp"])
if uploaded_file is not None:
    im = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8),cv2.IMREAD_COLOR)
    
    # Classificazione
    res = apply(im,mi)
    st.write(f"Nell'immagine appara: {res}")

    # ----
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    # Threshold of blue in HSV space
    lower_blue = np.array([hmin,smin,vmin])
    upper_blue = np.array([hmax,smax,vmax])
    # preparing the mask to overlay
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    result = cv2.bitwise_and(im, im, mask = mask)
    # ---
  
    st.image(result)
#------------

btn = st.button("Click me")
if btn:
    st.write("Ok ho cliccato")
  