# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 16:30:54 2018

@author: smd118
"""

import pandas as pd
import numpy as np

Output = 1

def load_data():
    #LOADING DATA
    glucData = pd.read_csv('GlucData3H_Overall.csv')
    glucData = glucData.drop(['Unnamed: 0', 'Operative', 't0', 'GF'], axis = 1)
    glucData['Gender'] = glucData['Gender'] == 'female'
    glucData['Gender'] = glucData['Gender'].astype(int)
    
    #LOGGING RELEVANT DATA
    glucData = glucData[glucData['SIt'] > 0]
    glucData = glucData[glucData['SIt+1'] > 0]
    glucData = glucData[glucData['SIt+2'] > 0]
    glucData = glucData[glucData['SIt+3'] > 0]
    
    glucData['Gt'] = np.log10(glucData['Gt'])
    glucData['SIt'] = np.log10(glucData['SIt'])
    glucData['SIt+' + str(Output)] = np.log10(glucData['SIt+' + str(Output)])
    
    glucData = glucData.reset_index()
    return glucData

def xval_gen(glucData, n):
    CrossValid =  [0, 0, 0, 0, 0]
    Patients = []
    for i in range(n):
        CrossValid[i] = np.zeros([0, 4])
        Patients.append([])
    q = 0
    for Patient, Data in glucData.groupby('Patient'):
        q = q+1
        CrossValid[q % n] = np.append(CrossValid[q % n], Data.loc[:,['SIt', 'Gt', 'SIt+' + str(Output), 'Sigma']].values, 0)
        Patients[q % n].append(Patient)
    return CrossValid, Patients

if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    glucData = load_data()
    sigma = np.load('Sigma_3D_' + str(Output) + 'H.npy')
    sigma = pd.DataFrame({'Sigma': sigma})
    glucData = pd.merge(glucData, sigma, left_index = True, right_index = True)
       
    res = 400
    grid_pts = [np.linspace(-8.5, -1.5, res), np.linspace(0.2, 1.4, res), np.linspace(-8.5, -1.5, res)] 
    
    for i in range(5):
        input_pts, patients = xval_gen(glucData, 5)
        input_pts[i] = np.zeros([0, 4])
        input_pts = np.concatenate(input_pts)