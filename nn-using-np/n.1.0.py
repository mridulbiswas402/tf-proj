import numpy as np
from tqdm import tqdm
from scipy.special import expit

# miscellaneous function. 

def threshold(x,cutoff=0.5):
  m=x.shape[0]
  res=[]
  for i in range(m):
    if(x[i]<cutoff):
      res.append(0)
    else:
      res.append(1)
  return np.array(res)      

def MSE(y,ypred):
  err=y-ypred
  err=err**2
  return sum(err)

def accuracy(y,ypred):
  count=0
  m=y.shape[0]
  for i in range(m):
    if y[i]==ypred[i]:
      count=count+1
  return count/m      

# implementation of a sigmoid neuron.

class neuron:
  def __init__(self,size):  # size is the size of vector x (i.e i/p vector).
    self.size=size
    self.weight=np.random.randn(self.size) # weight is randomly initialised vector of size same as x.
    self.bias=np.random.randn() # bias it is a real no.

  def feedforward(self,x):  # it is a sigmoid neuron. i.e y=sigmoid(w.x+b).
    z=np.dot(x,self.weight)+self.bias
    return expit(z)

  def fit(self,X_train,y_train,eta=0.1,epoch=10): # function for training the neuron's weights and bias.
    m=X_train.shape[0]  # m is no of training example used to train the neuron.
    error=[]
    for k in tqdm(range(epoch)):
      for i in range(m):
        index=np.random.randint(0,m-1) # randomly choosing a training example.
        ypred=self.feedforward(X_train[index]) # feeding it to neuron, ypred is the o/p of neuron.
        err=ypred*(ypred-y_train[index])*(1-ypred)
        grad=X_train[index]*err  # computing the gradient.
        self.weight=self.weight-eta*grad  # updating the weights.
        self.bias=self.bias-eta*err       # updating the bias.
      error.append(MSE(y_train,self.predict(X_train)))  # recording the avg error of neuron on the training set.
    return np.array(error)

  def predict(self,X):
    m=X.shape[0]
    ypred=[]
    for i in range(m):
      ypred.append(self.feedforward(X[i]))
    return np.array(ypred)



































