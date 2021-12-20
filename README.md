# Streamlit study app

* [App Link]( https://streamlit-app-example.herokuapp.com/)
*  https://streamlit-app-example.herokuapp.com/

## Setup

```
virtualenv env  # better virtualenv --python=python3.8 env
source env/bin/activate  
pip install -r requirements.txt  
# pip3 install streamlit opencv-python-headless onnxruntime 
streamlit run --server.enableCORS false --server.enableXsrfProtection false --server.port 8080  segmentation.py
```