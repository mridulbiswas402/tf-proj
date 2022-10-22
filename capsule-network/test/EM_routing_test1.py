import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
import matplotlib
import matplotlib.pyplot as plt
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

@tf.function
def safe_norm(s, axis=-1, epsilon=1e-7, keep_dims=False):
        squared_norm = tf.reduce_sum(tf.square(s),axis=axis,keepdims=keep_dims)
        return tf.sqrt(squared_norm + epsilon)

# data loading in appropriate formate

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension
x_train = x_train[..., tf.newaxis].astype("float64")
x_test = x_test[..., tf.newaxis].astype("float64")

X=x_train[:1]

X.shape

c1=tf.keras.layers.Conv2D(256,kernel_size=5,strides=1,padding='valid',activation='relu')
c2=tf.keras.layers.Conv2D(256,kernel_size=5,strides=2,padding='valid',activation='relu')
c3=tf.keras.layers.Conv2D(256,kernel_size=5,strides=2,padding='valid',activation='relu')
bn1=tf.keras.layers.BatchNormalization()
bn2=tf.keras.layers.BatchNormalization()

z=c3(bn2(c2(bn1(c1(X)))))

z.shape

z=tf.reshape(z,[-1,256,9]) # primary capsule.

z.shape

n=256
k=10
d=9
batch_size=3

z=tf.cast(z, tf.float64)

z=tf.tile(z,[batch_size,1,1]) # just get a batch of intermediate primary capsule on which EM algo will be applied

z.shape

# EM algorithm initialization step.

init_mu = random.rand(batch_size,k, d)*20 - 10
mu = init_mu  # mean initialization.

init_sigma = np.zeros((k, d, d))
for i in range(k):
    init_sigma[i] = np.eye(d)
sigma = init_sigma
sigma=tf.expand_dims(sigma,axis=0)
sigma=tf.tile(sigma,[batch_size,1,1,1]) # covariance matrix for a batch.

init_pi = np.ones(k)/k
pi = init_pi
pi=tf.expand_dims(pi,axis=0)
pi=tf.tile(pi,[batch_size,1]) 

R=np.zeros(shape=(n,k))
R=tf.expand_dims(R,axis=0)
R=tf.tile(R,[batch_size,1,1]) # coupling coefficient

pi=tf.Variable(pi,dtype=tf.float64)
mu=tf.Variable(mu,dtype=tf.float64)
sigma=tf.Variable(sigma,dtype=tf.float64)
R=tf.Variable(R,dtype=tf.float64)

print(mu.shape,pi.shape,sigma.shape,R.shape)

N=np.zeros((batch_size,n))
N=tf.Variable(N,dtype=tf.float64)

# E-step.
for i in range(k):
    for b in range(batch_size):
        tmp = tfp.distributions.MultivariateNormalFullCovariance(loc=mu[b][i],
                                                               covariance_matrix=sigma[b][i]).prob(z[b])
        N[b].assign(tmp)
    R[:,:,i].assign(tf.expand_dims(pi[:,i],axis=1)*N)
R.assign(R/tf.reduce_sum(R,axis=2, keepdims=True))

# M-step
N_k=tf.reduce_sum(R,axis=1)
pi=N_k/n
mu=tf.matmul(tf.transpose(R,perm=[0,2,1]),z)
mu=mu/N_k[:,:,None]

for i in range(k):
    tmp=z-tf.expand_dims(mu[:,i,:],axis=1)
    tmp=tf.expand_dims(tmp,axis=-1)
    tmp_T=tf.transpose(tmp,perm=[0,1,3,2])
    res=tf.matmul(tmp,tmp_T)
    res=tf.multiply(tf.reshape(R[:,:,i],[batch_size,n,1,1]),res)
    res=tf.reduce_sum(res,axis=1)/tf.reshape(N_k[:,i],[batch_size,1,1])
    sigma[:,i].assign(res)

sigma


### Rough work. ###.
N_k=tf.reduce_sum(R,axis=1)

N_k/n

mu=tf.matmul(tf.transpose(R,perm=[0,2,1]),z)

mu.shape

mu=mu/N_k[:,:,None]

mu[:,9,:]

tmp=z-tf.expand_dims(mu[:,i,:],axis=1)

tmp=tf.expand_dims(tmp,axis=-1)

tmp_T=tf.transpose(tmp,perm=[0,1,3,2])

tmp.shape

tmp_T.shape

res=tf.matmul(tmp,tmp_T)

res.shape

R.shape

res=tf.multiply(tf.reshape(R[:,:,i],[batch_size,n,1,1]),res)

res.shape

N_k

N_k[:,0]

tf.reduce_sum(res,axis=1)/tf.reshape(N_k[:,i],[batch_size,1,1])

sigma[:,i]





### rough ###

mu_tm=np.ones([3,3,2])

x_tm=np.ones([3,5,2])

x_tm-mu_tm[:,0,:].reshape([3,1,2])

