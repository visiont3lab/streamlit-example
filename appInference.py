import streamlit as st
import cv2
import numpy as np
from PIL import Image
from inference import modelInference
# https://www.onnxruntime.ai/python/auto_examples/plot_load_and_predict.html
# Requirements
# pip3 install streamlit opencv-python onnxruntime
# To run on replit type in the shell
# streamlit run --server.enableCORS false --server.enableXsrfProtection false --server.port 8080  appInference.py

def apply(im,mi):
  im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
  w = im.shape[1]
  h = im.shape[0]
  #print(w,h)
  x = cv2.resize(im, (224,224))
  x = x.astype(np.float32)/255.0
  x = np.transpose(x, (2, 0, 1) ) # BCHW
  x = x[np.newaxis,:,:,:]
  #x = np.random.random((1,3,224,224))
  res = mi.predict(x)
  xtl,ytl,xbr,ybr = tuple(res[0][0])
  #print(xtl,ytl,xbr,ybr)
  xtlPix,ytlPix,xbrPix,ybrPix =  ( int(xtl*w), int(ytl*h), int(xbr*w), int(ybr*h) ) 
  #print(xtlPix,ytlPix,xbrPix,ybrPix)
  im = cv2.rectangle(im, (xtlPix,ytlPix),(xbrPix,ybrPix),(0,255,0),2)
  return im

# Load Model 
@st.cache(suppress_st_warning=True)
def get_model():
  mi = modelInference(path2model="models/manuel.onnx")
  return mi
mi = get_model()

st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("Model Inference")
st.markdown('''
## Test dklsdsakfdd  Manuel Algoritmo

1. Load an Image [supported extension "png","jpeg","jpg","bmp"]
2. Run infernecedsa dsa
3. fd
''')
 
uploaded_file = st.file_uploader("Upload Image", type=["png","jpeg","jpg","bmp"])
if uploaded_file is not None:
  im = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8),cv2.IMREAD_COLOR)
  im = apply(im,mi)
  st.image(im, use_column_width=True ) # width=700)


run = st.button("Run Example Inference")
if run:
  im = cv2.imread("images/manuel.png",1)
  im = apply(im,mi)
  st.image(im, use_column_width=True ) # width=700)
