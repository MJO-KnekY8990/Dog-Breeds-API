import tensorflow as tf

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import Session
config = ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

import time

import keras

import streamlit as st

from PIL import Image

from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import numpy as np
import pandas as pd

st.set_option('deprecation.showfileUploaderEncoding', False)

@st.cache(persist=True)
def predict(img):
    #model = load_model('model')
    model = load_model('s3://mjo-dogs-breeds-mdl')
    labels = np.array(pd.read_csv('idx_to_class.csv')['0'])
    img = load_img('tst.jpg',target_size=(224,224))
    img = img_to_array(img)
    img = img.reshape((-1, img.shape[0], img.shape[1], 3)) / 255
    rez = model.predict(img)
    rez = pd.Series(rez[0][np.argpartition(rez,-3)[0][-3:]],labels[np.argpartition(rez,-3)[0][-3:]])
    rez = rez[rez>0.1]*100
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
        for i in range(rez.size):
            st.write('I am %.2f% sure that this is a %s.' % (rez[i], rez.index[i]))