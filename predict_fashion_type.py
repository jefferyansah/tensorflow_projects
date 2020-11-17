#%%
import tensorflow as tf 
import matplotlib.pyplot as plt 
import tensorflow as tf 
import numpy as np 
import tensorflow_datasets as tfds
import math
tfds.disable_progress_bar()
import logging
import os
import cv2
logger = tf.get_logger()
logger.setLevel(logging.ERROR)
import math
from scipy import ndimage
from tensorflow.keras.optimizers import SGD

def create_model():
    model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

    return model 

def create_cnn_model():
  model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=tf.nn.relu,
                           input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
                        ])
  model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
  return model


def normalize(images, labels):
  images = tf.cast(images, tf.float32)
  images /= 255
  return images, labels


def build_pipeline(checkpoint_path, network_type, epochs):

  dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
  train_dataset, test_dataset = dataset['train'], dataset['test']

  train_dataset =  train_dataset.map(normalize)
  test_dataset  =  test_dataset.map(normalize)

  if network_type == 'Basic Neural Nets':
    model = create_model()

  elif network_type == 'CNN':
    model = create_cnn_model()

  num_train_examples = metadata.splits['train'].num_examples
  num_test_examples = metadata.splits['test'].num_examples

  BATCH_SIZE = 32
  train_dataset = train_dataset.cache().repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
  test_dataset = test_dataset.cache().batch(BATCH_SIZE)

  #save model 
# Create a callback that saves the model's weights
  cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                  save_weights_only=True,
                                                  verbose=1)
  loss = model.fit(train_dataset,  
            epochs=epochs,
            steps_per_epoch=math.ceil(num_train_examples/BATCH_SIZE),
            callbacks=[cp_callback])

  test_loss, test_accuracy = model.evaluate(test_dataset, steps=math.ceil(num_test_examples/32))
  print('Accuracy on test dataset:', test_accuracy)
            
  return model, loss

image = cv2.imread('./boot.jpg', cv2.IMREAD_GRAYSCALE)

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
  return resized


def process_image(path, model_type):

  if model_type == 'CNN':
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    gray = cv2.resize(255-gray, (28, 28))
    flatten = gray.reshape(1,28,28,1)
  

  else: 
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    gray = cv2.resize(255-gray, (28, 28))
    (thresh, gray) = cv2.threshold(gray, 128, 255, 
      cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    flatten = gray.flatten() / 255.0 
    flatten = flatten.reshape((1,784))

  return  flatten,  gray


# def process_cnn_image(path, model_type):

#   if model_type == 'CNN':

#   gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#   gray = cv2.resize(255-gray, (28, 28))
#   (thresh, gray) = cv2.threshold(gray, 128, 255, 
#     cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#   flatten = gray.flatten() / 255.0 
#   flatten = flatten.reshape((1,784))

#   return  flatten,  gray

