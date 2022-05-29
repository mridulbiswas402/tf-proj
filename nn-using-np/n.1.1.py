import numpy as np
from tqdm import tqdm
from scipy.special import expit
from miscellaneous import *

# implementation of neural network.

class network:
  def __init__(self,archi): # archi is the architechture of the network it is list of on of node in each layer.
    self.layers=len(archi) # for example [4,3,2,2] => i/p size is 4 and 3,2 are no of neuron in hidden layer 1 and 2 respectively.
    self.archi=archi       # and 2 neuron in o/p layer.
    self.weight=[]  # list of weights of each layer. for our e.g: [w1(3x4),w2(2x3),w3(2x2)].   
    self.bias=[]    # list of bias of each layer for our e.g: [b1(3x1),b2(2x1),b3(2x1)].
    self.delta=[None]*(self.layers-1) # delta[i]= dJ/dz[i].
    self.z=[None]*(self.layers-1) # to store intermidiate results z[i]=W[i]*a[i-1]+b[i].
    self.a=[None]*(self.layers-1) # a[i]=sigmoid(z[i]).
    for i in range(self.layers-1): # initializing weights and biases randomly.
      self.weight.append(np.random.randn(archi[i+1],archi[i]))
      self.bias.append(np.random.randn(archi[i+1],1)) 

  def feedforward(self,x): # is the network.
    x=x.reshape(x.shape[0],1) 
    z=x
    for i in range(self.layers-1):
      z=expit(np.matmul(self.weight[i],z)+self.bias[i])
    return z 

  def forwardpass(self,x): # function for computing and storing the intermediate results.
    x=x.reshape(x.shape[0],1) 
    a=x
    for i in range(self.layers-1):
      z=np.matmul(self.weight[i],a)+self.bias[i] # z[i]=w[i]xa[i-1]+b[i] where a[0]=x.
      self.z[i]=z
      a=expit(z)                                 # a[i]=sigmoid(z[i]).
      self.a[i]=a
    return  


  def fit(self,X_train,y_train,eta=0.1,epoch=10): # for training the network's weights and biases.
    m=X_train.shape[0]
    error=[]
    for k in tqdm(range(epoch)):
      for i in range(m):
        index=np.random.randint(0,m-1) # randomly choosing a training example.
        self.forwardpass(X_train[index])  # computing and storing all intermediate results. 
        self.backprop(X_train[index],y_train[index],eta)  # computing dJ/dw[i] and dJ/db[i] and updating the weights and biases.
      error.append(MSE(y_train,self.predict(X_train))) # recording the avg error of network on the training set. 
    return np.array(error)

  def backprop(self,x,y,eta):
    #here layer are from r={0,1,...,self.layers-2} 1st layer is 0 and last layer is self.layers-2
    #i.e archi=[-1th,0th,1st,...,len(archi)-2] :: [1,2,...,len(archi)]
    r=self.layers-2
    y=y.reshape(self.a[r].shape)
    x=x.reshape(x.shape[0],1)

    #1. computing delta[r]::delta[layers-2]
    self.delta[r]=(self.a[r]-y)*sigmaprime(self.z[r])
    r=r-1
    # computing remaining delta[1 to r-1] :: delta[0 to layers-2]
    while(r>=0):
      self.delta[r]=(np.matmul(self.weight[r+1].T,self.delta[r+1]))*sigmaprime(self.z[r])
      r=r-1

    # udating weights
    grad=np.matmul(self.delta[0],x.T)
    self.weight[0]=self.weight[0]-eta*grad
    self.bias[0]=self.bias[0]-eta*self.delta[0]
    for i in range(1,self.layers-1):
      grad=np.matmul(self.delta[i],self.a[i-1].T) # dJ/dw[i]=delta[i]xa[i-1].T
      self.weight[i]=self.weight[i]-eta*grad      # updating weights and biases.
      self.bias[i]=self.bias[i]-eta*self.delta[i]

    return

  def predict(self,X):
    m=X.shape[0]
    ypred=[]
    for i in range(m):
      ypred.append(self.feedforward(X[i]))
    return np.array(ypred)




















