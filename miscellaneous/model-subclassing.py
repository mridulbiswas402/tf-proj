import numpy as np
import tensorflow as tf
from tensorflow import keras


# data loading in appropriate formate

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension
x_train = x_train[..., tf.newaxis].astype("float32")
x_test = x_test[..., tf.newaxis].astype("float32")


#building model using subclassing

class MyModel(tf.keras.Model):

  def __init__(self):
    super(MyModel,self).__init__()
    self.flatten = keras.layers.Flatten(input_shape=(28, 28))
    self.d1 = keras.layers.Dense(128, activation='relu')
    self.d2 = keras.layers.Dense(10)

  def call(self, inputs):
    x=self.flatten(inputs)
    x = self.d1(x)
    return self.d2(x)

model = MyModel()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# training model

model.fit(x_train, y_train, batch_size=128,epochs=5,validation_split=0.2)