import streamlit as st
import numpy as np
import matplotlib.pyplot as plt 
from convertor import *
from predict_fashion_type import *
import pandas as pd
from PIL import Image, ImageOps 
import cv2
import math
from scipy import ndimage


checkpoint_path = 'training_1/cp.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
# initialize the dimensions of the image to be resized and
# grab the image size
  dim = None
  (h, w) = image.shape[:2]
  # if both the width and height are None, then return the
  # original image
  if width is None and height is None:
      return image
  # check to see if the width is None
  if width is None:
      # calculate the ratio of the height and construct the
      # dimensions
      r = height / float(h)
      dim = (int(w * r), height)
  # otherwise, the height is None
  else:
      # calculate the ratio of the width and construct the
      # dimensions
      r = width / float(w)
      dim = (width, int(h * r))
  # resize the image
  resized = cv2.resize(image, dim, interpolation = inter)
  # return the resized image
  st.write(resized.shape)
  return resized

def transform_image(resized_image):
    resized_image = resized_image/255
    return resized_image.reshape((1,784))


st.sidebar.write('TENSORFLOW PROJECTS')

select_project = st.sidebar.selectbox('Select Project', 
            ['Tempreture Convertor', 'Fashion Predictor'])

if select_project == 'Tempreture Convertor':

    st.write('Celcius(°C) to Farhenreit (°F) Convertor 🌡️using TensorFlow')
    st.write('Model Details')
    with st.spinner('Training the Neural Network'):
        loss, model = build_model()
    st.line_chart(loss.history['loss'])
    user_input = st.text_input('Enter a Value in Degrees Celcius', 0)
    run = st.button('Predict')
    user_input = float(user_input)
    if run:
        farh_value = model.predict([user_input]).flatten().tolist()[0]
        st.write('The Predict Value of ', user_input , '°C in °F is', round(farh_value,2) )
        st.write(model.summary())

if select_project == 'Fashion Predictor':

    network_type = st.sidebar.selectbox('Select Neural Network Type',
                    ('Basic Neural Nets', 'CNN'))

    labels = ['T-shirt/top', 'Trouser', 'Pullover', 
                 'Dress', 'Coat', 'Sandal', 'Shirt', 
                 'Sneaker', 'Bag', 'Ankle boot']
    label_dict = {}
    for label in labels:
        add_label = {labels.index(label): label}
        label_dict.update(add_label)
    
    st.sidebar.write('Configure Model Parameters')
    training_epochs = st.sidebar.slider('Training Epochs', 1, 10, 1)

    retrain_model = st.sidebar.button('Retrain Neural Network')

    if retrain_model:
        with st.spinner('Retraining the Neural Network'):
            if network_type == 'CNN':
                checkpoint_path = 'cnn_training_1/cp.ckpt'
                checkpoint_dir = os.path.dirname(checkpoint_path)
                model, loss = build_pipeline(checkpoint_path,
                     network_type, training_epochs)
                st.line_chart(loss.history['loss'])
                st.sidebar.write('Model Retrained, Weights Saved for Use')
            else:
                checkpoint_path = 'basic_training_1/cp.ckpt'
                checkpoint_dir = os.path.dirname(checkpoint_path)
                model, loss = build_pipeline(checkpoint_path, 
                                             network_type,
                                             training_epochs)
                st.line_chart(loss.history['loss'])
                st.sidebar.write('Model Retrained, Weights Saved for Use')

    upload_image = st.sidebar.file_uploader('Upload Image')
    if upload_image is not None:
        image = Image.open(upload_image) #.convert('LA')
        st.image(image, width=600)
        if image.mode == 'P':
            image = image.convert('RGB')
        filename = 'file.' + image.format
        # if image.format == 'JPEG' or image.format == 'JPG':
        #     filename = 'file.' + 'PNG'
        # else:
        #     filename = 'file.' + image.format
        image_array = np.array(image)
        cv2.imwrite(filename, cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))

        predict_button = st.button('Predict Fashion Type')
        if predict_button:

            if network_type == 'CNN':
                model = create_cnn_model()
                checkpoint_path = 'cnn_training_1/cp.ckpt'
                model.load_weights(checkpoint_path)
                model_input, im_array  = process_image(filename, network_type)
                st.write(model_input.shape)
                prediction = model.predict([model_input])

            if network_type == 'Basic Neural Nets':
                model = create_model()
                checkpoint_path = 'basic_training_1/cp.ckpt'
                model.load_weights(checkpoint_path)
                model_input, im_array  = process_image(filename, network_type)
                prediction = model.predict([model_input])

            
            confidence  = (prediction * 100).flatten().tolist()[np.argmax(prediction)]
            st.write('Predicted Class: ', np.argmax(prediction))
            st.write('Class label: My model thinks this is a ', label_dict[np.argmax(prediction)], 'with confidence ', confidence)

            predict_prob = {'Labels': labels, 'Probabilities (%)': (prediction* 100).flatten().tolist() }
            data = pd.DataFrame.from_dict(predict_prob)
            st.write(data)
            plt.figure()
            plt.imshow(im_array, cmap=plt.cm.binary)
            plt.colorbar()
            plt.grid(False)
            plt.show()
            st.pyplot(plt)

    


    
