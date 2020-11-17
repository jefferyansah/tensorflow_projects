#%%
import matplotlib.pyplot as plt 
import tensorflow as tf 
import numpy as np 
import tensorflow_datasets as tfds
import math
tfds.disable_progress_bar()
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

#
#%%
tfds.list_builders()
# %%
#load the datasets
dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

#%%
class_names = metadata.features['label'].names
print("Class names: {}".format(class_names))
# %%
num_train_examples = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples
print("Number of training examples: {}".format(num_train_examples))
print("Number of test examples:     {}".format(num_test_examples))

# %%
def normalize(images, labels):
  images = tf.cast(images, tf.float32)
  images /= 255
  return images, labels

# The map function applies the normalize function to each element in the train
# and test datasets
train_dataset =  train_dataset.map(normalize)
test_dataset  =  test_dataset.map(normalize)

# The first time you use the dataset, the images will be loaded from disk
# Caching will keep them in memory, making training faster
train_dataset =  train_dataset.cache()
test_dataset  =  test_dataset.cache()
# %%
for image, label in test_dataset.take(1):
    break
image = image.numpy().reshape((28,28))

# plt.figure()
# plt.imshow(image, cmap=plt.cm.binary)
# plt.colorbar()
# plt.grid(False)
# plt.show()
# %%
plt.figure(figsize=(10,10))
for i, (image, label) in enumerate(test_dataset.take(45)):
    image = image.numpy().reshape((28,28))
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image, cmap=plt.cm.binary)
    plt.xlabel(class_names[label])
plt.show()
# %%
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

#%%
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# %%
BATCH_SIZE = 32
train_dataset = train_dataset.cache().repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
test_dataset = test_dataset.cache().batch(BATCH_SIZE)

#%%
model.fit(train_dataset, epochs=5, steps_per_epoch=math.ceil(num_train_examples/BATCH_SIZE))
#%%
test_loss, test_accuracy = model.evaluate(test_dataset, steps=math.ceil(num_test_examples/32))
print('Accuracy on test dataset:', test_accuracy)
# %%
def plot_image(i, predictions_array, true_labels, images):
  predictions_array, true_label, img = predictions_array[i], true_labels[i], images[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img[...,0], cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
  
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')
# %%
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions, test_labels)
# %%
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
# %%
img = test_images[0]
img = np.array([img])
#print(img.shape)

predictions_single = model.predict(img)

print(img)
# %%
plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
# %%
np.argmax(predictions_single[0])
# %%
model.summary()
# %%
from PIL import Image
import numpy as np
import tensorflow as tf 
image = Image.open("./passport_photo.jpg")
image = Image.open('./passport_photo.jpg').convert('LA')
a = tf.keras.preprocessing.image.img_to_array(image)/255
img = tf.keras.preprocessing.image.load_img(
    './passport_photo.jpg', target_size=(28, 28))
img_array = tf.keras.preprocessing.image.img_to_array(img) /255
img_array.shape
# %%
img = keras.preprocessing.image.load_img(
    './passport_photo.jpg')
img_array = keras.preprocessing.image.img_to_array(img)/255
img_array.shape
# %%
import cv2 
img = cv2.imread('./passport_photo.jpg')
cv2.imshow('Heya', img )
# %%
from PIL import Image
import numpy as np
import math
image = Image.open("./passport_photo.jpg")
image = Image.open('./passport_photo.jpg').convert('LA')
image
# %%
import cv2
import matplotlib.pyplot as plt
import math
from scipy import ndimage
# create an array where we can store our 4 pictures

def process_image(image):
gray = cv2.imread('./out.jpeg', cv2.IMREAD_GRAYSCALE)
gray = cv2.resize(255-gray, (28, 28))
while np.sum(gray[0]) == 0:
    gray = gray[1:]

while np.sum(gray[:,0]) == 0:
    gray = np.delete(gray,0,1)

while np.sum(gray[-1]) == 0:
    gray = gray[:-1]

while np.sum(gray[:,-1]) == 0:
    gray = np.delete(gray,-1,1)

rows,cols = gray.shape

if rows > cols:
    factor = 28.0/rows
    rows = 28
    cols = int(round(cols*factor))
    gray = cv2.resize(gray, (cols,rows))
else:
    factor = 28/cols
    cols = 28
    rows = int(round(rows*factor))
    gray = cv2.resize(gray, (cols, rows))

colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')
def getBestShift(img):
    cy,cx = ndimage.measurements.center_of_mass(img)

    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx,shifty
def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted
gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')
shiftx,shifty = getBestShift(gray)
shifted = shift(gray,shiftx,shifty)
gray = shifted
gray = cv2.resize(255-gray, (28, 28))
(thresh, gray) = cv2.threshold(gray, 128, 255, 
  cv2.THRESH_BINARY | cv2.THRESH_OTSU)
flatten = gray.flatten() / 255.0
plt.figure()
plt.imshow(gray, cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
plt.show()
# %%
plt.figure()
plt.imshow(gray, cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
plt.show()

# %%
flatten.shape
# %%
from PIL import Image
import numpy as np
import math
image = Image.open("./tensorflow_projects/coat.png")
image
# %%
