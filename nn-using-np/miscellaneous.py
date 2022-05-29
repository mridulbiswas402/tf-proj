import numpy as np
from scipy.special import expit

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
  ypred=ypred.reshape(y.shape)
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

def sigmaprime(x):
  z=expit(x)
  return (z*(1-z))
