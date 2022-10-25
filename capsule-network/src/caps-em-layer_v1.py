import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
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

        batch_size = input_shape[0]
        n=input_shape[1]
        k=self.caps_n
        d=self.caps_dim

        self.W = self.add_weight(name='W',
                    shape=[1, input_shape[1], self.caps_n, self.caps_dim, input_shape[-1]],
                    dtype=tf.float64,
                    initializer='glorot_uniform',
                    trainable=True)
        
        #initialization step.
        init_mu = random.rand(batch_size,k, d)*20 - 10
        self.mu = init_mu #initializing mean.

        init_sigma = np.zeros((k, d, d))
        for i in range(k):
            init_sigma[i] = np.eye(d)
        sigma = init_sigma
        sigma=tf.expand_dims(sigma,axis=0)
        self.sigma=tf.tile(sigma,[batch_size,1,1,1]) # initializing cov matrix.

        init_pi = np.ones(k)/k
        pi = init_pi
        pi=tf.expand_dims(pi,axis=0)
        self.pi=tf.tile(pi,[batch_size,1])

        R=np.zeros(shape=(n,k))
        R=tf.expand_dims(R,axis=0)
        self.R=tf.tile(R,[batch_size,1,1]) # coupling coefficient.
        
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
        """#initialization step.
        init_mu = random.rand(batch_size,k, d)*20 - 10
        mu = init_mu #initializing mean.

        init_sigma = np.zeros((k, d, d))
        for i in range(k):
            init_sigma[i] = np.eye(d)
        sigma = init_sigma
        sigma=tf.expand_dims(sigma,axis=0)
        sigma=tf.tile(sigma,[batch_size,1,1,1]) # initializing cov matrix.

        init_pi = np.ones(k)/k
        pi = init_pi
        pi=tf.expand_dims(pi,axis=0)
        pi=tf.tile(pi,[batch_size,1])

        R=np.zeros(shape=(n,k))
        R=tf.expand_dims(R,axis=0)
        R=tf.tile(R,[batch_size,1,1]) # coupling coefficient."""

        pi=tf.Variable(self.pi,dtype=tf.float64)
        mu=tf.Variable(self.mu,dtype=tf.float64)
        sigma=tf.Variable(self.sigma,dtype=tf.float64)
        R=tf.Variable(self.R,dtype=tf.float64)

        #print(mu.shape,pi.shape,sigma.shape,R.shape)

        N=np.zeros((batch_size,n))
        N=tf.Variable(N,dtype=tf.float64)

        r=self.r
        while(r):
          r=r-1
          # E-step.
          for i in range(k):
              for b in range(batch_size):
                  tmp = tfp.distributions.MultivariateNormalFullCovariance(loc=mu[b][i],
                                                                        covariance_matrix=sigma[b][i]).prob(input_tensor[b])
                  N[b].assign(tmp)
              R[:,:,i].assign(tf.expand_dims(pi[:,i],axis=1)*N)
          R.assign(R/tf.reduce_sum(R,axis=2, keepdims=True))

          # M-step
          N_k=tf.reduce_sum(R,axis=1)
          pi=N_k/n
          mu=tf.matmul(tf.transpose(R,perm=[0,2,1]),input_tensor)
          mu=mu/N_k[:,:,None]

          for i in range(k):
              tmp=input_tensor-tf.expand_dims(mu[:,i,:],axis=1)
              tmp=tf.expand_dims(tmp,axis=-1)
              tmp_T=tf.transpose(tmp,perm=[0,1,3,2])
              res=tf.matmul(tmp,tmp_T)
              res=tf.multiply(tf.reshape(R[:,:,i],[batch_size,n,1,1]),res)
              res=tf.reduce_sum(res,axis=1)/tf.reshape(N_k[:,i],[batch_size,1,1])
              sigma[:,i].assign(res)
              
        weighted_prediction=tf.multiply(caps_predicted,tf.reshape(R,[batch_size,n,k,1,1]))
        weighted_sum = tf.reduce_sum(weighted_prediction, axis=1, keepdims=True)
        v=squash(weighted_sum, axis=-2)
        v = tf.squeeze(v, axis=[1,4])
        return v

    def compute_output_signature(self,input_shape):
      return tf.TensorSpec(shape=[input_shape[0],self.caps_n,self.caps_dim],dtype=tf.float64)