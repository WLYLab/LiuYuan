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
from sklearn import metrics
from performance import performance
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
    print(train_seq[1])
    train_seq=one_hot(train_seq)
    test_seq=one_hot(test_seq)
    return train_seq,test_seq,train_label,test_label


# In[4]:


train_data='D.melanogaster_train.csv'
test_data='D.melanogaster_test.csv'
out_name='D.melanogaster'
train_seq,test_seq,train_label_,test_label_=load_data(train_data,test_data)
train_label=to_categorical(np.array(train_label_))
test_label=to_categorical(np.array(test_label_))

# In[5]:


model=ResnetBuilder.build_resnet_18((41, 4), 2)
adam=Adam()
model.compile(loss='binary_crossentropy',optimizer=adam,metrics=['accuracy'])


# In[6]:


early_stopping = EarlyStopping(monitor='val_loss', patience=6, verbose=1, mode='auto')
checkpointer = ModelCheckpoint(
            filepath=out_name+'_best_model',
            verbose=1, save_best_only=True, mode='auto', period=1)


# In[ ]:


# model.fit(train_seq,train_label,
#     batch_size=64,
#     validation_split=0.15,
#     epochs=200,
#     verbose=1,
#     shuffle=True,callbacks=[early_stopping,checkpointer])



# In[ ]:

attention=AttentionLayer()
test_model=load_model(out_name+'_best_model',custom_objects={'AttentionLayer':attention})
loss,acc=test_model.evaluate(test_seq,test_label)
print('loss:',loss)
print('acc:',acc)
out=test_model.predict(test_seq)
pred_proba=out[:,1]
pred=np.argmax(out,axis=1)
pred=np.argmax(pred_proba)
pred=np.array([1 if x>0.5 else 0 for x in pred_proba])

acc=metrics.accuracy_score(test_label_,pred)
print(acc)
roc_auc=metrics.roc_auc_score(test_label_,pred_proba)
MCC=metrics.matthews_corrcoef(test_label_, pred)
precision, recall, SN, SP, GM, TP, TN, FP, FN = performance(test_label_, pred)
performance_result={
    'acc':[acc],
    'roc_auc':[roc_auc],
    'precision':[precision],
    'recall':[recall],
    'SN':[SN],
    'SP':[SP],
    'GM':[GM],
    'MCC':[MCC],
    'TP':[TP],
    'FP':[FP],
    'TN':[TN],
    'FN':[FN]
}
result={
            'label':test_label.tolist(),
            'pred':pred.tolist(),
            'pred_proba':pred_proba.tolist()
        }
pd.DataFrame(performance_result).to_csv(out_name+'_result.csv',index=False)

pd.DataFrame(result).to_csv(out_name+'_predict.csv',index=False)


