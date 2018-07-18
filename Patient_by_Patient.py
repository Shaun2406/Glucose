# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 15:47:19 2018

@author: smd118
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
from multiprocessing import Pool, Lock, Array
plt.close("all")

glucData = pd.read_csv('GlucDataOverall.csv')
glucData= glucData.loc[:,['Patient', 'SIt', 'SIt+1', 'Gt']]

#LOGGING RELEVANT DATA
glucData = glucData[glucData['SIt'] > 0]
glucData = glucData[glucData['SIt+1'] > 0]

glucData['Gt'] = np.log10(glucData['Gt'])
glucData['SIt+1'] = np.log10(glucData['SIt+1'])
glucData['SIt'] = np.log10(glucData['SIt'])

glucData = glucData.reset_index()

for Patient, Data in glucData.groupby('Patient'):
    print(Patient)
    X = Data['Gt'].values
    print(Data['Gt'].values)
    
    
    
    
    
    
    
    
    np.save(Patient,X)
    break