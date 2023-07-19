#!/usr/bin/env python
# coding: utf-8

# In[11]:


import joblib
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')


features_select=pd.read_csv('C:/Users/Cola/Desktop/MIMIC1/selection_feature.csv')



sme = SMOTE(random_state=42)

X=features_select.iloc[:,2:]#x的范围是第一列到最后一列
y=features_select["target"]#括号内是最后一列的名称
X_train, X_test, y_train, y_test = train_test_split(features_select, features_select['target'], test_size=0.2, random_state=82, stratify=features_select.target)
x_train = X_train.iloc[:, 2:]
x_test = X_test.iloc[:, 2:]
x_bsm, y_bsm = sme.fit_resample(x_train, y_train)



clf = XGBClassifier(colsample_bytree=0.3, gamma=0.01, learning_rate=0.1, max_depth=20, n_estimators=300)
clf.fit(x_bsm, y_bsm)


pickle.dump(clf,open("clf.dat","wb"))
joblib.dump(clf, 'model.pkl')
