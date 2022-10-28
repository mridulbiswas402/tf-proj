""" This is tensorflow implementation of EM algorithm
this version supports parallel batch processing """

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import matplotlib.pyplot as plt
from scipy import random
from scipy.stats import multivariate_normal

def gen_data(k=3, dim=2, points_per_cluster=200, lim=[-10, 10]):
    '''
    Generates data from a random mixture of Gaussians in a given range.
    Will also plot the points in case of 2D.
    input:
        - k: Number of Gaussian clusters
        - dim: Dimension of generated points
        - points_per_cluster: Number of points to be generated for each cluster
        - lim: Range of mean values
    output:
        - X: Generated points (points_per_cluster*k, dim)
    '''
    x = []
    mean = random.rand(k, dim)*(lim[1]-lim[0]) + lim[0]
    for i in range(k):
        cov = random.rand(dim, dim+10)
        cov = np.matmul(cov, cov.T)
        _x = np.random.multivariate_normal(mean[i], cov, points_per_cluster)
        x += list(_x)
    x = np.array(x)
    if(dim == 2):
        fig = plt.figure()
        ax = fig.gca()
        ax.scatter(x[:,0], x[:,1], s=3, alpha=0.4)
        ax.autoscale(enable=True) 
    return x

x1 = gen_data(k=3, dim=2, points_per_cluster=100)
x2 = gen_data(k=3, dim=2, points_per_cluster=100)
x3 = gen_data(k=3, dim=2, points_per_cluster=100)
x4 = gen_data(k=3, dim=2, points_per_cluster=100)

X=np.array([x1,x2,x3,x4])

x=tf.convert_to_tensor(X,dtype=tf.float64)

x.shape

k=3 # on of clusters.
d=x.shape[-1] # dim of each data point
n=x.shape[1] # cardinality of data points.
b=x.shape[0] # batch_size

print(k,d,n,b)

x.shape # [b,n,d]

# parameter initialization for batch of data.

pi=np.array([[0.3333,0.3333,0.3333],
             [0.3333,0.3333,0.3333],
             [0.3333,0.3333,0.3333],
             [0.3333,0.3333,0.3333]])

mu=np.array([[x[0][6],x[0][100],x[0][46]],
             [x[1][6],x[1][100],x[1][46]],
             [x[2][6],x[2][100],x[2][46]],
             [x[3][6],x[3][100],x[3][46]]])

sigma=np.array([[[[1.0,0],[0,1]],[[1,0],[0,1]],[[1,0],[0,1]]],
                [[[1.0,0],[0,1]],[[1,0],[0,1]],[[1,0],[0,1]]],
                [[[1.0,0],[0,1]],[[1,0],[0,1]],[[1,0],[0,1]]],
                [[[1.0,0],[0,1]],[[1,0],[0,1]],[[1,0],[0,1]]]])

R=np.zeros(shape=(b,n,k))


pi=tf.Variable(pi,dtype=tf.float64)
mu=tf.Variable(mu,dtype=tf.float64)
sigma=tf.Variable(sigma,dtype=tf.float64)
R=tf.Variable(R,dtype=tf.float64)

print(pi.shape,mu.shape,sigma.shape,R.shape)

"""-- E-step. --"""

x_tmp=tf.expand_dims(x,axis=1) # x.shape==[b,n,d]
x_tmp=tf.tile(x_tmp,[1,k,1,1]) # x_tmp.shape==[b,k,n,d]

mu_tmp=tf.expand_dims(mu,axis=2) # mu.shape==[b,k,d]
mu_tmp=tf.tile(mu_tmp,[1,1,n,1])   # mu_tmp.shape==[b,k,n,d]

sig_tmp=tf.expand_dims(sigma,axis=2) # sigma.shape==[b,k,d,d]
sig_tmp=tf.tile(sig_tmp,[1,1,n,1,1])   # sig_tmp.shape == [b,k,n,d,d]

#print(x_tmp.shape,mu_tmp.shape,sig_tmp.shape)

N = tfp.distributions.MultivariateNormalFullCovariance(loc=mu_tmp,covariance_matrix=sig_tmp).prob(x_tmp)
#print(N.shape)
N = pi[:,:,None]*N
N = N/tf.expand_dims(tf.reduce_sum(N,axis=1),axis=1)
R = tf.transpose(N,perm=[0,2,1])

"""-- M-step. --"""

# updating pi.
N_k = tf.reduce_sum(R,axis=1)
pi = N_k/n

# updating mu.
mu = tf.matmul(tf.transpose(R,perm=[0,2,1]),x)
mu = mu/N_k[:,:,None]

# updating sigma.

mu_tmp=tf.expand_dims(mu,axis=2)
mu_tmp=tf.tile(mu_tmp,[1,1,n,1])

x_tmp=x_tmp-mu_tmp
x_tmp=tf.reshape(x_tmp,[b,k,n,d,1])
x_tmp_T=tf.transpose(x_tmp,perm=[0,1,2,4,3])
res = tf.matmul(x_tmp,x_tmp_T)

R_T=tf.transpose(R,perm=[0,2,1])

res = tf.multiply(tf.reshape(R_T,[b,k,n,1,1]),res)
sigma = tf.reduce_sum(res,axis=2)/tf.reshape(N_k,[b,k,1,1])

