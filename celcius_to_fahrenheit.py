
import numpy as np 
import tensorflow as tf
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)
import matplotlib.pyplot as plt 


# Create sample dataset
celsius_q    = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)
fahrenheit_a = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)

for i,c in enumerate(celsius_q):

    print("{} degrees Celcius = {} degrees faherenheit".format(c, fahrenheit_a[i]))


#build the Layer for the Neural  Networl
l0 = tf.keras.layers.Dense(units= 1, input_shape = [1]) # Layer
model = tf.keras.Sequential([l0]) # Assembling the Layer

### Alternatively put it in a one-liner
model = tf.keras.Sequential([tf.keras.layers.Dense(units= 1, input_shape = [1])])

#compile the model: by specifying the loss and then optimzation function
model.compile(loss = 'mean_squared_error',
             optimizer=tf.keras.optimizers.Adam(0.1))

#fit the model to the data
history = model.fit(celsius_q, fahrenheit_a, epochs=5000, verbose=False)
print('Model Training Completed')


#plot the Loss for the Training epochs
plt.xlabel('Epoch Number')
plt.ylabel('Loss Magnitude')
plt.plot(history.history['loss'])


# test the model on unseen data
print(model.predict([0, 304.1]))


#Inspect the layers
print('These are the layer variables: {}'.format(l0.get_weights()))


#Try thesame  data example by on multilayered neural network
l0 = tf.keras.layers.Dense(units=4, input_shape=[1])
l1 = tf.keras.layers.Dense(units=4)
l2 = tf.keras.layers.Dense(units=1)
model = tf.keras.Sequential([l0, l1, l2])
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))
model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
print("Finished training the model")
print(model.predict([100.0]))
print("Model predicts that 100 degrees Celsius is: {} degrees Fahrenheit".format(model.predict([100.0])))
print("These are the l0 variables: {}".format(l0.get_weights()))
print("These are the l1 variables: {}".format(l1.get_weights()))
print("These are the l2 variables: {}".format(l2.get_weights()))