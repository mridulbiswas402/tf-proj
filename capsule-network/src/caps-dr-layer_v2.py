import numpy as np
import tensorflow as tf
from tensorflow import keras

# %matplotlib inline
import matplotlib
import matplotlib.pyplot as plt

import os
import time

@tf.function
def squash(x, axis=-1):
    s_squared_norm = tf.math.reduce_sum(tf.math.square(x), axis, keepdims=True) + keras.backend.epsilon()
    scale = tf.math.sqrt(s_squared_norm) / (1 + s_squared_norm)
    return scale * x

@tf.function
def margin_loss(y_true, y_pred):
    lamb, margin = 0.5, 0.1
    return tf.math.reduce_sum((y_true * tf.math.square(tf.nn.relu(1 - margin - y_pred)) + lamb * (
        1 - y_true) * tf.math.square(tf.nn.relu(y_pred - margin))), axis=-1)

#@tf.function
def safe_norm(s, axis=-1, epsilon=1e-7, keep_dims=False):
        squared_norm = tf.reduce_sum(tf.square(s),axis=axis,keepdims=keep_dims)
        return tf.sqrt(squared_norm + epsilon)
    


class Capsule(keras.layers.Layer):
   

    def __init__(self,
                 num_capsule,
                 dim_capsule,
                 routings=3,
                 **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.caps_n = num_capsule
        self.caps_dim = dim_capsule
        self.r = routings

    def get_config(self):
        config = super().get_config().copy()
        config.update({
        'num_capsule':  self.caps_n,
        'dim_capsule' : self.caps_dim,
        'routings':  self.r,      
        })
        return config

    def build(self, input_shape):
        
        self.W = self.add_weight(name='W',
                    shape=[1, input_shape[1], self.caps_n, self.caps_dim, input_shape[-1]],
                    dtype=tf.float32,
                    initializer='glorot_uniform',
                    trainable=True)

    def call(self, input_tensor):
        batch_size = input_tensor.shape[0]
        
        W_tiled = tf.tile(self.W, [batch_size, 1, 1, 1, 1]) # replicating the weights for parallel processing of a batch.
        """ W_tiled.shape=[batch_size,caps_n(i-1),caps_n(i),caps_dim(i),caps_dim(i-1)] """

        caps_output_expanded = tf.expand_dims(input_tensor, -1) # converting last dim to a column vector.
        """ the above step change the input shape from 
            [batch_size,caps_n(i-1),caps_dim(i-1)] --> [batch_size,caps_n(i-1),caps_dim(i-1),1]"""

        caps_output_tile = tf.expand_dims(caps_output_expanded, 2)
        """ the above step change the input shape from 
            [batch_size,caps_n(i-1),caps_dim(i-1),1] --> [batch_size,caps_n(i-1),1,caps_dim(i-1),1]"""

        caps_output_tiled = tf.tile(caps_output_tile, [1, 1, self.caps_n, 1, 1]) # replicating the input capsule vector for every output capsule.
        """ i.e [batch_size,caps_n(i-1),1,caps_dim(i-1),1] --> [batch_size,caps_n(i-1),caps_n(i),1,caps_dim(i-1),1]"""

        caps_predicted = tf.matmul(W_tiled, caps_output_tiled) # this is performing element wise tf.matmul() operation.
        """ caps_predicted.shape = [1,caps_n(i-1),caps_n(i),caps_dim(i),1]"""

        """ dynamic routing """
        raw_weights = tf.zeros([batch_size,input_tensor.shape[1] , self.caps_n, 1, 1]) # non trainable weights.
        """ raw_weights.shape=[batch_size,caps_n(i-1) ,caps_n(i), 1, 1]"""

        r=self.r
        while(r):
          r-=1
          routing_weights = tf.nn.softmax(raw_weights,axis=2)
          """ [batch_size,caps_n(i-1) ,caps_n(i), 1, 1]  softmax applied along the pointed dim.
                                           ^                                                   """

          weighted_predictions = tf.multiply(routing_weights, caps_predicted)
          """ weighted_predictions.shape = [batch_size, caps_n(i-1),caps_n(i),caps_dim(i), 1]"""

          weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keepdims=True)
          """ [batch_size,caps_n(i-1) ,caps_n(i),caps_dim(i), 1]  sum applied along the pointed dim.
                               ^                                                               
          therefore weighted_sum.shape=[batch_size,1 ,caps_n(i),caps_dim(i), 1]"""

          v = squash(weighted_sum, axis=-2) #normalize to unit length vector.
          v_tiled = tf.tile(v, [1, input_tensor.shape[1], 1, 1, 1])
          """ v_tiled.shape=[batch_size,caps_n(i-1),caps_n(i),caps_dim(i), 1]"""

          agreement = tf.matmul(caps_predicted, v_tiled,transpose_a=True)
          """ agreement.shape=[batch_size,caps_n(i-1),caps_n(i), 1, 1]"""

          if(r>0):
              routing_weights+=agreement
          else:
              v = tf.squeeze(v, axis=[1,4])
              return v
    
    def compute_output_signature(self,input_shape):
        return tf.TensorSpec(shape=[input_shape[0],self.caps_n,self.caps_dim],dtype=tf.float32)