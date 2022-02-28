#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 16:03:16 2022

@author: benseimon
"""

#%% Packages

import pandas as pd
import numpy as np
import os 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors 
from sklearn.model_selection import train_test_split
from utils.helper_functions import *
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, roc_curve, auc, confusion_matrix 
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler

#%% Loading data

#locally 
local_path = '/Users/benseimon/Documents/Barca GSE/Studies/Term 2/CML2/Project 1/In-Class'

#g drive
from google.colab import drive
drive.mount('/content/drive')
drive_path = '/content/drive/MyDrive/CML_2_Projects/Project 1/In-Class'

#%% Model

#train-test split 

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=3, test_size=0.33)


#scale

# scale continuous variables
scaler = StandardScaler()
features_to_scale = []
scaler.fit(X_train[features_to_scale])
X_train_scaled[features_to_scale] = scaler.transform(X_train[features_to_scale])
X_test_scaled[features_to_scale] = scaler.transform(X_test[features_to_scale])


# model 

:k_vals, 'weights':weights}
grid_knn = GridSearchCV(MyKNN, param_grid=grid_values, scoring='accuracy', cv=5) 
grid_knn.fit(X_train, y_train)

print('best parameters:', grid_knn.best_params_)
print('best score:', grid_knn.best_score_)

y_pred = grid_knn.predict(X_test)
y_prob_pred = grid_knn.predict_proba(X_test)

#Results
cm = confusion_matrix(y_test, y_pred_res)
plot_confusion_matrix(cm, ['Other','Target'])

## In-sample
print(roc_auc_score(y_train, grid_knn_acc.predict_proba(X_train)[:, 1]))

## In-sample
roc_auc_score(y_test, grid_knn_acc.predict_proba(X_test)[:, 1])




