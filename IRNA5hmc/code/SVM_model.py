# !/use/bin/env python
# encoding:utf-8
import pandas as pd
import numpy as np
import itertools
from sklearn.model_selection import KFold  
from sklearn import svm
from sklearn.model_selection import train_test_split
import math
# from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
import easy_excel
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import *
import sklearn.ensemble
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import sys
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB  
import subprocess
from sklearn.utils import shuffle
import itertools
from sklearn.ensemble import GradientBoostingClassifier
import sys
path=""
inputname=sys.argv[1]
outputname=inputname.split('.')[0]
crossvalidation_values=int(sys.argv[2])
CPU_values=int(sys.argv[3])
name=outputname

def performance(labelArr, predictArr):
    #labelArr[i] is actual value,predictArr[i] is predict value
    TP = 0.; TN = 0.; FP = 0.; FN = 0.
    for i in range(len(labelArr)):
        if labelArr[i] == 1 and predictArr[i] == 1:
            TP += 1.
        if labelArr[i] == 1 and predictArr[i] == 0:
            FN += 1.
        if labelArr[i] == 0 and predictArr[i] == 1:
            FP += 1.
        if labelArr[i] == 0 and predictArr[i] == 0:
            TN += 1.
    SN = TP/(TP + FN) #Sensitivity = TP/P  and P = TP + FN
    SP = TN/(FP + TN) #Specificity = TN/N  and N = TN + FP
    precision=TP/(TP+FP)
    recall=TP/(TP+FN)
    GM=math.sqrt(recall*SP)
    #MCC = (TP*TN-FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    return precision,recall,SN,SP,GM,TP,TN,FP,FN

    
if __name__=="__main__":
    # In[ ]:
    
    """
        cross validation and f-score and xgboost
    """
    datapath =path+outputname+".csv"
    classifier="SVM"
    mode="crossvalidation"
    #print "start"
    train_data = pd.read_csv(datapath, header=None, index_col=None)
    X = np.array(train_data)
    Y = list(map(lambda x: 1, xrange(len(train_data) // 2)))
    Y2 = list(map(lambda x: 0, xrange(len(train_data) // 2)))
    Y.extend(Y2)
    Y = np.array(Y)
    svc = svm.SVC(probability=True)
    parameters = {'kernel': ['rbf'], 'C': [math.pow(2,e) for e in range(-5,15,2)], 'gamma': [math.pow(2,e) for e in range(-15, -5, 2)]}
    #parameters = {'kernel': ['rbf'], 'C':map(lambda x:2**x,np.linspace(-2,5,7)), 'gamma':map(lambda x:2**x,np.linspace(-5,2,7))}
    clf = GridSearchCV(svc, parameters, cv=crossvalidation_values, n_jobs=CPU_values, scoring='accuracy')
    clf.fit(X, Y)
    C=clf.best_params_['C']
    gamma=clf.best_params_['gamma']
    y_predict=cross_val_predict(svm.SVC(kernel='rbf',C=C,gamma=gamma),X,Y,cv=crossvalidation_values,n_jobs=CPU_values)
    y_predict_prob=cross_val_predict(svm.SVC(kernel='rbf',C=C,gamma=gamma,probability=True),X,Y,cv=crossvalidation_values,n_jobs=CPU_values,method='predict_proba')
    joblib.dump(clf,path+classifier+mode+outputname+".model")
    predict_save=[Y.astype(int),y_predict.astype(int),y_predict_prob[:,1]]
    predict_save=np.array(predict_save).T
    pd.DataFrame(predict_save).to_csv('Before_'+path+classifier+mode+outputname+"_"+'_predict_crossvalidation.csv',header=None,index=False)
    ROC_AUC_area=metrics.roc_auc_score(Y,y_predict_prob[:,1])
    ACC=metrics.accuracy_score(Y,y_predict)
    precision, recall, SN, SP, GM, TP, TN, FP, FN = performance(Y, y_predict)
    F1_Score=metrics.f1_score(Y, y_predict)
    F_measure=F1_Score
    MCC=metrics.matthews_corrcoef(Y, y_predict)
    pos=TP+FN
    neg=FP+TN
    savedata=[[['SVM'+"C:"+str(C)+"gamma:"+str(gamma),ACC,precision, recall,SN, SP, GM,F_measure,F1_Score,MCC,ROC_AUC_area,TP,FN,FP,TN,pos,neg]]]
    easy_excel.save(classifier+"_crossvalidation",[str(X.shape[1])],savedata,path+'cross_validation_'+classifier+"_"+outputname+'.xls')
