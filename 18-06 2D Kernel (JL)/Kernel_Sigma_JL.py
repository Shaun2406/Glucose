# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 12:06:41 2018

Non-parallel calculation of probability field, as well as calculation of sigma matrix
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats
plt.close("all")

#LOADING DATA
GlucData = pd.read_csv('C:\WinPython-64bit-3.5.4.1Qt5\Glucose\GlucDataOverall.csv')
GlucData = GlucData.drop(['Unnamed: 0', 'Unnamed: 0.1', 'Operative', 'Patient', 't0', 'GF'], axis = 1)
GlucData['Gender'] = GlucData['Gender'] == 'female'
GlucData['Gender'] = GlucData['Gender'].astype(int)
GlucData = GlucData[GlucData['SIt'] > 0]
GlucData = GlucData[GlucData['SIt+1'] > 0]

#Create an Ortho-Normalised Matrix Xdec - 2D
X = GlucData.loc[:,['SIt', 'SIt+1']].values
C = np.cov(np.transpose(X))
R = np.linalg.cholesky(C)
A = np.linalg.inv(np.transpose(R))
detA = np.linalg.det(A)
Xmean = np.mean(X, 0)
X0 = X - Xmean
Xdec = np.matmul(X0, A)

#Scaling Factors from Root Matrix (X), Standard Deviation and Max Range - 2D
Rad = np.max(np.linalg.norm(Xdec, axis = 1))
X_std = np.std(X,0)
X_iqr = scipy.stats.iqr(X,0)/1.348
S_X = np.min([X_std, X_iqr], 0)

k = len(X)
Sigma = np.zeros([k, 2])
for i in range(k):
    if i % 1000 == 0:
        print(i)
    m = np.linalg.norm(Xdec-Xdec[i,:], axis = 1)
    m = m < k**(-1/6)
    Sigma[i] = (np.sum(m)*Rad**2*k**(1/3))**(-1/6)*S_X
    
np.save('Sigma_JL', Sigma)