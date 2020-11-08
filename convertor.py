
import numpy as np 
import tensorflow as tf
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)
import matplotlib.pyplot as plt 
import streamlit as st


celsius_q    = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)
fahrenheit_a = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)

def build_model():
    l0 = tf.keras.layers.Dense(units=4, input_shape=[1])
    l1 = tf.keras.layers.Dense(units=4)
    l2 = tf.keras.layers.Dense(units=1)
    model = tf.keras.Sequential([l0, l1, l2])
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))
    training = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
    return training, model

