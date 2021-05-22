# -*- coding: utf-8 -*-
"""
Created on Sat May 22 18:34:47 2021

@author: JPadariya
"""

import numpy as np
import pandas as pd
filename =r'C:\Users\jpadariya\Downloads\assignment\assignment\res\final.csv'
dff = pd.read_csv(filename, encoding= 'unicode_escape')
dff.rename(columns={"": "cng"})
dff['class']='0'

for inde,i in enumerate(dff['car_model']):
    if int(dff['length'][inde]) < 4000  or (int(dff['total_seats'][inde]) < 5) :
        dff['class'][inde] = '1'
    if int(dff['length'][inde]) > 4000 or (int(dff['total_seats'][inde]) > 5):
        dff['class'][inde] = '2'


s  = dff['functionality'].str.replace("'",'').str.split(',').explode().to_frame()

cols = s['functionality'].drop_duplicates(keep='first').tolist()

df2 = pd.concat([dff, pd.crosstab(s.index, s["functionality"])[cols]], axis=1).replace(
    {1: True, 0: False}
)
print(df2)
df2=df2.rename(columns={"": "cng"})
   

df2.to_csv(r"C:\Users\jpadariya\Downloads\assignment\assignment\res\training_file.csv")

#%%

conditions = [
    (dff['length'] < 4) & (dff['total_seats'] < 5),
    (dff['length'] > 4) and (dff['total_seats'] > 5),
    ]


# create a list of the values we want to assign for each condition
values = ['1', '2']
dff['class']='0'
# create a new column and use np.select to assign values to it using our lists as arguments
dff['class'] = np.select(conditions, values)

# display updated DataFrame
dff.head()
    
#%%
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.metrics import mean_squared_error
#import libraries
from datetime import datetime, timedelta,date
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_predict
%matplotlib inline
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from __future__ import division
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
#do not show warnings
import warnings
warnings.filterwarnings("ignore")

#import machine learning related libraries
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import pickle
import joblib


df2=pd.read_csv(r"C:\Users\jpadariya\Downloads\assignment\assignment\res\training_file.csv",encoding= 'unicode_escape')

y= df2['class']


X=df2[df2.columns[3:]]


X=X.astype(int)
# s = np.array(byte_list)
# X = np.frombuffer(s, dtype=np.uint8)
# y = np.frombuffer(s, dtype='S1')
# X, y

# XX = np.reshape(X, (-1, 1))
# yy =np.reshape(y, (-1, 1))



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)

#create an array of models
models = []
models.append(("LR",LogisticRegression()))
models.append(("NB",GaussianNB()))
models.append(("RF",RandomForestClassifier()))
models.append(("SVC",SVC()))
models.append(("Dtree",DecisionTreeClassifier()))
models.append(("XGB",xgb.XGBClassifier()))
models.append(("KNN",KNeighborsClassifier()))


for name,model in models:
    kfold = KFold(n_splits=4, random_state=2)
    cv_result = cross_val_score(model,X_train,y_train, cv = kfold,scoring = "accuracy")
    print(name, cv_result)
    y_pred = cross_val_predict(model, X, y, cv=5)
    
    # with open("models_{}.pckl".format(name), "wb") as f:
    #   pickle.dump(model, f)
#    filename = 'a1q4-{}.sav'.format(name)
#    joblib.dump(model, filename)
    print(confusion_matrix(y, y_pred))
    print(accuracy_score(y, y_pred))
    

 
    # this_column = df.columns[i]
    # df[this_column] = [i, i+1]
    
    
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, y_pred)

######accuracy scores######

"""
LR [0.71428571 0.85714286 0.85714286 0.42857143]
0.6857142857142857

NB [0.71428571 0.85714286 1.         0.57142857]
0.7428571428571429

RF [0.85714286 0.85714286 1.         0.85714286]
0.8571428571428571


SVC [0.57142857 0.85714286 1.         0.28571429]
0.6

Dtree [1.         1.         1.         0.85714286]
0.9428571428571428

Xgb
0.8857142857142857


KNN [0.71428571 1.         1.         0.28571429]
0.8
"""
