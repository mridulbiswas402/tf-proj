"""template code for generating appropriate directory structure need to create tf.datasets using keras api for 
generating tf.datasets from directory structure."""

from random import seed
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split


# path of source dirs.

covid='./COVID-19_Radiography_Dataset/COVID/images'
normal='./COVID-19_Radiography_Dataset/Normal/images'
pneumonia='./COVID-19_Radiography_Dataset/Viral Pneumonia/images'

# paths of target dirs.

trcov='./CXRDATA/Train/COVID'
trnor='./CXRDATA/Train/Normal'
trpneu='./CXRDATA/Train/Pneumonia'
tscov='./CXRDATA/Test/COVID'
tsnor='./CXRDATA/Test/Normal'
tspneu='./CXRDATA/Test/Pneumonia'

data_dirs=[covid,normal,pneumonia]
train_dir=[trcov,trnor,trpneu]
test_dir=[tscov,tsnor,tspneu]

#data=[]

for i in range(len(data_dirs)):
    temp=os.listdir(data_dirs[i])
    #data.append(temp)  
    df=pd.DataFrame(temp,columns=['path'])
    train, test = train_test_split(df,shuffle=True,test_size=0.2)
    for k in range(len(train)):
        os.rename(data_dirs[i]+'/'+train.iloc[k][0],train_dir[i]+'/'+train.iloc[k][0])
    for k in range(len(test)):
        os.rename(data_dirs[i]+'/'+test.iloc[k][0],test_dir[i]+'/'+test.iloc[k][0])


#print(base+covid+'/'+train.iloc[0][0],trcov+'/'+train.iloc[0][0])
