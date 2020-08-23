import tensorflow as tf

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import Session
config = ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

import time

from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import Input

import keras

import streamlit as st

from PIL import Image

import re

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from keras.applications.imagenet_utils import decode_predictions

import numpy as np
import pandas as pd

from keras import backend as K
import keras.backend.tensorflow_backend as tfback

st.set_option('deprecation.showfileUploaderEncoding', False)

@st.cache(persist=True)
def predict(img):
    model = InceptionResNetV2(weights="imagenet",
                  include_top=True,
                  input_tensor=Input(shape=(224, 224, 3)))
    img = img_to_array(img)
    img = img.reshape((-1, img.shape[0], img.shape[1], 3)) / 255
    rez = model.predict(img)
    rez = pd.DataFrame(decode_predictions(rez)[0],columns=['_','breed','prct']).drop(columns='_').sort_values(by='prct', ascending=False)
    np.array(rez)
    return rez
    
st.title("Dog Breed Classifier\nCNN ResNetV2")

uploaded_file = st.file_uploader("Upload your dog picture")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Picture', use_column_width=True)
    st.write("")
    st.write("...Identifying...")
    with st.spinner("...Sorting results..."):
        time.sleep(5)
        rez = predict(image)
        for i in range(len(rez)):
            if rez[i][1]>0.1:
                st.write('I am %.2f% sure that this is a %s.' % (rez[i][1]*100, re.sub('_', ' ',rez[i][0])))