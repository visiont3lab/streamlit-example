import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Requirements
# pip3 install streamlit opencv-python 
# To run on replit type in the shell
# streamlit run --server.enableCORS false --server.port 8080  --server.maxUploadSize=500 appInference.py

st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("Model Inference")

st.markdown('''

## Inference Model Application

## Setup

### Local PC

```pyhton
virtualen env
source env/bin/activate
pip install streamlit opencv-python
```

### Setup Colab

!wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
!unzip ngrok-stable-linux-amd64.zip
get_ipython().system_raw('./ngrok http 8501 &')
!curl -s http://localhost:4040/api/tunnels | python3 -c \
    'import sys, json; print("Execute the next cell and the go to the following URL: " +json.load(sys.stdin)["tunnels"][0]["public_url"])'
!pkill -9 ngrok
''')


st.markdown('''
## Demo Algoritmo

1. Load an Image [supported extension "png","jpeg","jpg","bmp"]
2. Run infernece
''')
 
uploaded_file = st.file_uploader("Upload Image", type=["png","jpeg","jpg","bmp"])
if uploaded_file is not None:
  print("Jew")
  #print(np.fromstring(uploaded_file.read(), np.uint8))
  image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8),cv2.IMREAD_COLOR)

  #image = Image.open(uploaded_file)
  #base64_img_bytes = uploaded_file.read() # byte
  #decoded_image_data = base64.decodebytes(base64_img_bytes)
  #nparr = np.fromstring(decoded_image_data, np.uint8)
  #img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # cv2.IMREAD_COLOR in OpenCV

  # ----------
  # Inference Here

  #------------
  st.image(image, use_column_width=True ) # width=700)


run = st.button("Run Inference")
if run:
  st.text("Someone clicked me")