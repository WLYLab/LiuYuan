#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
from keras.models import Sequential,Model,load_model
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras import backend as K
from ResNet1D import ResnetBuilder
import tensorflow as tf
import keras
from Attention import AttentionLayer
from one_hot_encode import one_hot
import pandas as pd
from keras.utils import to_categorical
from keras.optimizers import Adam
from Attention import AttentionLayer

# In[2]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# In[3]:


#加载数据
def load_data(train_data,test_data):
    Train=pd.read_csv(train_data, sep=',', header=0)
    Test=pd.read_csv(test_data, sep=',', header=0)
    
    train_seq=Train['sequence'].values.tolist()
    test_seq=Test['sequence'].values.tolist()
    
    train_label=Train['label'].values.tolist()
    test_label=Test['label'].values.tolist()
    train_seq=one_hot(train_seq)
    test_seq=one_hot(test_seq)
    train_label=to_categorical(np.array(train_label))
    test_label=to_categorical(np.array(test_label))
    
    return train_seq,test_seq,train_label,test_label


# In[4]:


train_data='A.thaliana_train.csv'
test_data='A.thaliana_test.csv'
train_seq,test_seq,train_label,test_label=load_data(train_data,test_data)
attention=AttentionLayer()
test_model=load_model('save_model_path',custom_objects={'AttentionLayer':attention})
acc,loss=test_model.evaluate(test_seq,test_label)
print(acc)
print(loss)
