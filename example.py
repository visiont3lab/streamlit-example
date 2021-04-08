import streamlit as st 
import pandas as pd
import numpy as np
import cv2

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
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    # ----
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

age = st.slider('How old are you?', 0, 130, 25)
st.write("I'm ", age, 'years old')
  