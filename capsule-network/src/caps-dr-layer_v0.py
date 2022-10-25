import numpy as np
import tensorflow as tf
from tensorflow import keras

from keras.engine import data_adapter

@tf.function
def squash(v,epsilon=1e-7,axis=-1):
    sqnrm=tf.reduce_sum(tf.square(v), axis=axis,keepdims=True)
    nrm=tf.sqrt(sqnrm + epsilon) #safe norm to avoid divide by zero.
    sqsh_factor = sqnrm / (1. + sqnrm)
    unit_vect = v / nrm
    return sqsh_factor*unit_vect

@tf.function
def safe_norm(s, axis=-1, epsilon=1e-7, keep_dims=False):
        squared_norm = tf.reduce_sum(tf.square(s),axis=axis,keepdims=keep_dims)
        return tf.sqrt(squared_norm + epsilon)



"""-- layers --"""

class Primary_caps_layer(tf.keras.layers.Layer):
  """ caps_n(i) --> no of capsule in ith layer 
      caps_dim(i) --> dimension of capsule in ith layer. 
      
      primary_caps_layer output shape = [batch_size,caps_n,caps_dim]"""

  def __init__(self,caps_dim=8,caps_n=1152):
    super(Primary_caps_layer, self).__init__()
    self.caps_n=caps_n  # no of capsule in this layer.
    self.caps_dim=caps_dim # dim of each capsule in this layer
    self.conv1=tf.keras.layers.Conv2D(256,kernel_size=9,strides=1,padding='valid',activation='relu') #@ changes may be needed of no of kernel.
    self.conv2=tf.keras.layers.Conv2D(256,kernel_size=9,strides=2,padding='valid',activation='relu')

  def call(self, input_tensor):
    x=self.conv1(input_tensor)
    x=self.conv2(x)
    x=tf.reshape(x,[-1,self.caps_n,self.caps_dim])
    return squash(x)


class Digit_caps_layer(tf.keras.layers.Layer):
  """ caps_n(i) --> no of capsule in ith layer 
      caps_dim(i) --> dimension of capsule in ith layer. 
      and we assume this is ith layer. 
      output.shape of ith layer = [batch_size, 1,caps_n(i),caps_dim(i), 1]"""

  def __init__(self,caps_dim=16,caps_n=10,r=3):
    super(Digit_caps_layer,self).__init__()
    self.caps_n=caps_n # no of capsule.
    self.caps_dim=caps_dim # dim of each capsule.
    self.r=r # no of iteration in routing by agreement algorithm.
  
  def build(self,input_shape): # input_shape = [batch_size,caps_n(i-1),caps_dim(i-1)] 
    self.W = tf.Variable(initial_value=tf.random.normal(
    shape=(1, input_shape[1], self.caps_n, self.caps_dim, input_shape[-1]),
    stddev=0.1, dtype=tf.float32),
    trainable=True)  #weigth initialization for this layer W.shape=[1,caps_n(i-1),caps_n(i),caps_dim(i),caps_dim(i-1)].

  def call(self,input_tensor): #input_tensor.shape=[batch_size,caps_n(i-1),caps_dim(i-1)]
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
    " i.e [batch_size,caps_n(i-1),1,caps_dim(i-1),1] --> [batch_size,caps_n(i-1),caps_n(i),1,caps_dim(i-1),1]"

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
          return v


"""-- network --"""
class Caps_net(tf.keras.Model):

  def __init__(self,no_classes=10):
    super(Caps_net,self).__init__()
    self.pri_layer=Primary_caps_layer(caps_dim=8,caps_n=1152)
    self.dig_layer=Digit_caps_layer(caps_dim=16,caps_n=10,r=3)

  def call(self,input_tensor):
    x = self.pri_layer(input_tensor) #x.shape=[batch_size,caps_n(i),caps_dim(i)]
    x = self.dig_layer(x) #x.shape=[batch_size, 1,caps_n(i),caps_dim(i), 1]

    """The lengths of the output vectors represent the class probabilities, 
       so we could just use tf.norm() to compute them,"""
    x = safe_norm(x, axis=-2) #x.shape=[batch_size,1,caps_n(i-1),1]

    x = tf.nn.softmax(x,axis=2) #converting those probabilities to prob dist.
    x = tf.squeeze(x, axis=[1,3]) #reducing the extra dims. therefore the output shape =[batch_size,caps_n(i-1)] 
    return x