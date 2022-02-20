#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 17:05:44 2022

@author: benseimon
"""

#%% import packages and read in data

import pandas as pd
import numpy as np 
import os
import plotly.express as px



data_path = '/Users/benseimon/Documents/Barca GSE/Studies/Term 2/CML2/Project 1/Data'
os.chdir(data_path)
comorbidities = pd.read_csv('MIMIC_diagnoses.csv')
diagnosis_definitions = pd.read_csv('MIMIC_metadata_diagnose.csv')
#feature_definitions = pd.read_csv('mimic_patient_metadata.xlsx')
train_data = pd.read_csv('mimic_train.csv')
test_data = pd.read_csv('mimic_test_death.csv')


#drop as per instructions 
train_data = train_data.drop(['DOD', 'DISCHTIME', 'DEATHTIME', 'LOS'], axis = 1)

#cumulative chart 
diagnosis_cumulative = np.cumsum(train_data['ICD9_diagnosis'].value_counts(normalize=True).sort_values(ascending=False))
px.area(
x=range(1, diagnosis_cumulative.shape[0]+1), 
y = diagnosis_cumulative,
labels={"x": "diagnosis", "y": "Proportion of patients"})

deaths = train_data.groupby('ICD9_diagnosis').agg({'ICD9_diagnosis': 'count','HOSPITAL_EXPIRE_FLAG': 'sum'}).sort_values(by = 'HOSPITAL_EXPIRE_FLAG', ascending = False).reset_index()
deaths = deaths.rename(columns = {'HOSPITAL_EXPIRE_FLAG': 'Total_Deaths', 'ICD9_diagnosis': 'Total_Occurrences'})
deaths = deaths.reset_index()
deaths = deaths.rename(columns = {'ICD9_diagnosis': 'ICD9_CODE'})

deaths = pd.merge(deaths,diagnosis_definitions[['ICD9_CODE','SHORT_DIAGNOSE']],on='ICD9_CODE', how='left')
