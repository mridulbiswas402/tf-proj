import numpy as np
import tensorflow as tf
from tensorflow import keras


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

class Primary_caps_layer(tf.keras.layers.Layer):
  """ caps_n(i) --> no of capsule in ith layer 
      caps_dim(i) --> dimension of capsule in ith layer. 
      
      primary_caps_layer output shape = [batch_size,caps_n,caps_dim]"""

  def __init__(self,caps_n=1152,k1=256,k2=256,k_s1=9,k_s2=5,s1=1,s2=3):
    super(Primary_caps_layer, self).__init__()
    self.caps_n=caps_n  # no of capsule in this layer.(as initialized by usr this may be changed based on other parameters.)
    self.k1=k1          # no of filter in 1st conv layer.
    self.k2=k2          # no of filter in 2nd conv layer.
    self.k_s1=k_s1      # kernel_size of 1st conv layer.
    self.k_s2=k_s2      # kernel_size of 2nd conv layer.
    self.s1=s1          # stride in 1st conv layer.
    self.s2=s2          # stride in 2nd conv layer.
    self.conv1=tf.keras.layers.Conv2D(k1,kernel_size=k_s1,strides=s1,padding='valid',activation='relu') 
    self.conv2=tf.keras.layers.Conv2D(k2,kernel_size=k_s2,strides=s2,padding='valid',activation='relu')

  def call(self, input_tensor):
    batch_size=input_tensor.shape[0]
    x=self.conv1(input_tensor)
    x=self.conv2(x) 

    assert x.shape[1]*x.shape[1]*self.k2==self.caps_n*self.caps_dim # $ eqn--1

    x=tf.reshape(x,[batch_size,self.caps_n,self.caps_dim]) # *
    return squash(x)

  def build(self,input_shape):
    self.batch_size=input_shape[0] 
    tmp=int(((input_shape[1]-self.k_s1)/self.s1))+1
    self.conv1_output_shape=[input_shape[0],tmp,tmp,self.k1]
    tmp=int(((tmp-self.k_s2)/self.s2))+1
    self.conv2_output_shape=[input_shape[0],tmp,tmp,self.k2]
    tmp1=tmp*tmp*self.k2
    self.caps_n=self.caps_n-(tmp1%self.caps_n) # recomputing apropriate no of capsule : $ eqn--1 is true.
    self.caps_dim=int((tmp*tmp*self.k2)/self.caps_n); # same is done for caps_dim.
    

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
          return v
