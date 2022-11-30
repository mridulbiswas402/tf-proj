import numpy as np
import tensorflow as tf
from tensorflow import keras


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


        
"""
routing-less-capsule-network with fully connected pose weigths + routing weights.

"""

class Capsule(keras.layers.Layer):
   

    def __init__(self,
                 num_capsule,
                 dim_capsule,
                 **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.caps_n = num_capsule
        self.caps_dim = dim_capsule

    def get_config(self):
        config = super().get_config().copy()
        config.update({
        'num_capsule':  self.caps_n,
        'dim_capsule' : self.caps_dim,    
        })
        return config

    def build(self, input_shape):

        self.W = self.add_weight(name='W',
                    shape=[1, input_shape[1], self.caps_n, self.caps_dim, input_shape[-1]],
                    dtype=tf.float32,
                    initializer='glorot_uniform',
                    trainable=True)
        
        self.R = self.add_weight(name='R',
                    shape=[1, input_shape[1], self.caps_n],
                    dtype=tf.float32,
                    initializer='glorot_uniform',
                    trainable=True)
        
        
    def call(self, input_tensor):
        batch_size = input_tensor.shape[0]
        n=input_tensor.shape[1]
        k=self.caps_n
        
        W_tiled = tf.tile(self.W, [batch_size, 1, 1, 1, 1])
        
        R_tiled = tf.tile(self.R,[batch_size,1,1])
        R_tiled = tf.nn.softmax(R_tiled,axis=1)

        caps_output_expanded = tf.expand_dims(input_tensor, -1) # converting last dim to a column vector.
        caps_output_tile = tf.expand_dims(caps_output_expanded, 2)
        caps_output_tiled = tf.tile(caps_output_tile, [1, 1, self.caps_n, 1, 1]) # replicating the input capsule vector for every output capsule.
        caps_predicted = tf.matmul(W_tiled, caps_output_tiled) # this is performing element wise tf.matmul() operation.       
        weighted_prediction=tf.multiply(caps_predicted,tf.reshape(R_tiled,[batch_size,n,k,1,1]))
        weighted_sum = tf.reduce_sum(weighted_prediction, axis=1, keepdims=True)
        v=squash(weighted_sum, axis=-2)
        v = tf.squeeze(v, axis=[1,4])
        return v

    def compute_output_signature(self,input_shape):
      return tf.TensorSpec(shape=[input_shape[0],self.caps_n,self.caps_dim],dtype=tf.float32)
