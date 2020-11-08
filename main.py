import streamlit as st
import numpy as np
import matplotlib.pyplot as plt 
from convertor import *


st.sidebar.write('Celcius(°C) to Farhenreit (°F) Convertor 🌡️using TensorFlow')


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