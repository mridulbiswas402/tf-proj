import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from scipy import random

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
                    dtype=tf.float64,
                    initializer='glorot_uniform',
                    trainable=True)
        
        
    def call(self, input_tensor):
        assert input_tensor.shape[2]==self.caps_dim
        input_tensor=tf.cast(input_tensor,dtype=tf.float64)
        assert input_tensor.dtype==tf.float64
        batch_size = input_tensor.shape[0]
        n=input_tensor.shape[1]
        k=self.caps_n
        d=self.caps_dim
        
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
        #initialization step.
        
        pi=np.ones([batch_size,k])/k
        mu=np.random.rand(batch_size,k,d)
        sigma=np.ones([batch_size,k,d])
        R=np.zeros(shape=(batch_size,n,k))

        pi=tf.convert_to_tensor(pi,dtype=tf.float64)
        mu=tf.convert_to_tensor(mu,dtype=tf.float64)
        sigma=tf.convert_to_tensor(sigma,dtype=tf.float64)
        R=tf.convert_to_tensor(R,dtype=tf.float64)

        r=self.r
        while(r):
          r=r-1
          """ E-step. """
          
          x_tmp=tf.expand_dims(input_tensor,axis=1) # x.shape==[b,n,d]
          x_tmp=tf.tile(x_tmp,[1,k,1,1]) # x_tmp.shape==[b,k,n,d]

          mu_tmp=tf.expand_dims(mu,axis=2) # mu.shape==[b,k,d]
          mu_tmp=tf.tile(mu_tmp,[1,1,n,1])   # mu_tmp.shape==[b,k,n,d]

          sig_tmp=tf.expand_dims(sigma,axis=2) # sigma.shape==[b,k,d]
          sig_tmp=tf.tile(sig_tmp,[1,1,n,1])   # sig_tmp.shape == [b,k,n,d]

          N = tfd.MultivariateNormalDiag(loc=mu_tmp,scale_diag=sig_tmp).prob(x_tmp)
          N = pi[:,:,None]*N
          N = N/tf.expand_dims(tf.reduce_sum(N,axis=1),axis=1)
          R = tf.transpose(N,perm=[0,2,1])

          """ M-step. """
          
          # updating pi.
          N_k = tf.reduce_sum(R,axis=1)
          pi = N_k/n

          # updating mu.
          mu = tf.matmul(tf.transpose(R,perm=[0,2,1]),input_tensor)
          mu = mu/N_k[:,:,None]

          # updating sigma.
          mu_tmp=tf.expand_dims(mu,axis=2)
          mu_tmp=tf.tile(mu_tmp,[1,1,n,1])
          x_tmp=x_tmp-mu_tmp
          x_tmp=tf.square(x_tmp)
          R_T=tf.transpose(R,perm=[0,2,1])
          x_tmp = tf.multiply(tf.reshape(R_T,[batch_size,k,n,1]),x_tmp)
          sigma = tf.reduce_sum(x_tmp,axis=2)/tf.reshape(N_k,[batch_size,k,1])
          sigma=tf.sqrt(sigma)
              
        weighted_prediction=tf.multiply(caps_predicted,tf.reshape(R,[batch_size,n,k,1,1]))
        weighted_sum = tf.reduce_sum(weighted_prediction, axis=1, keepdims=True)
        v=squash(weighted_sum, axis=-2)
        v = tf.squeeze(v, axis=[1,4])
        return v

    def compute_output_signature(self,input_shape):
      return tf.TensorSpec(shape=[input_shape[0],self.caps_n,self.caps_dim],dtype=tf.float64)