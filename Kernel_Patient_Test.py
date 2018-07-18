# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 16:15:21 2018

@author: smd118
"""

import pandas as pd
import numpy as np
from time import time
from multiprocessing import Pool, Array
import matplotlib.pyplot as plt
import scipy.integrate
plt.close("all")

def load_data():
    #LOADING DATA
    glucData = pd.read_csv('GlucDataOverall.csv')
    glucData = glucData.drop(['Unnamed: 0', 'Unnamed: 0.1', 'Operative', 't0', 'GF'], axis = 1)
    glucData['Gender'] = glucData['Gender'] == 'female'
    glucData['Gender'] = glucData['Gender'].astype(int)
    
    #LOGGING RELEVANT DATA
    glucData = glucData[glucData['SIt'] > 0]
    glucData = glucData[glucData['SIt+1'] > 0]
    
    glucData['Gt'] = np.log10(glucData['Gt'])
    glucData['SIt+1'] = np.log10(glucData['SIt+1'])
    glucData['SIt'] = np.log10(glucData['SIt'])
    
    glucData = glucData.reset_index()
    return glucData

def xval_gen(glucData, n):
    CrossValid =  [0, 0, 0, 0, 0]
    for i in range(n):
        CrossValid[i] = np.zeros([0, 4])
    q = 0
    for Patient, Data in glucData.groupby('Patient'):
        q = q+1
        CrossValid[q % n] = np.append(CrossValid[q % n], Data.loc[:,['SIt', 'Gt', 'SIt+1', 'index']].values, 0 )
    return CrossValid

def Interpolate(idx, tgt):
    if linear_prob[idx] < tgt:
        inter = grid_pts[idx+2]
    else:
        inter = grid_pts[idx+1]
    return inter

if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    
    res = 150
    
    grid_pts = np.linspace(-8.5, -1.5, res)    
    glucData = load_data()
    glucData.insert(19, 'SIt+1_UB', np.zeros(len(glucData)))
    xvalData = xval_gen(glucData, 5)
    current_patient = np.zeros([1,1])
    q = 0
    '''for i in range(5):
        conf_pdf = np.load('PDF_3D_' + str(i+1) + '.npy')
        test_pts = xvalData[i]
        Conf_Int = np.zeros(len(test_pts))
        for j in range(len(test_pts)):
            linear_prob = scipy.integrate.cumtrapz(conf_pdf[j,:], grid_pts)/np.trapz(conf_pdf[j,:], grid_pts)
            Upper_Bound = 10**Interpolate(np.argmin(abs(linear_prob - 0.95)), 0.95)
            glucData.loc[glucData['index'] == test_pts[j,3], 'SIt+1_UB'] = Upper_Bound
            #if current_patient[0][0] != glucData.loc[glucData['index'] == test_pts[j,3],['Patient']].values:
            #    q = q+1
            #    if q % 50 == 0:
            #        print(q)
            #    current_patient = glucData.loc[glucData['index'] == test_pts[j,3],['Patient']].values
            #    patient_data = glucData.loc[glucData['Patient'] == current_patient[0][0]]
glucData.to_csv('GlucDataOverall_UB.csv')'''            
glucData = pd.read_csv('GlucDataOverall_UB.csv')
print(np.sum(glucData['SIt+1_UB'] < 10**glucData['SIt+1'])/62589*100)