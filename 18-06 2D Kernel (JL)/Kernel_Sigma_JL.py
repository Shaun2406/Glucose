# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 12:06:41 2018

Calculation of Sigma matrix for JL kernel method 
"""

import pandas as pd
import numpy as np
import scipy.stats

#LOADING DATA
GlucData = pd.read_csv('GlucDataOverall.csv')
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
Rad = np.sort(np.linalg.norm(Xdec, axis = 1))
Rad = Rad[round(len(Rad)*0.95)]
X_std = np.std(X,0)
X_iqr = scipy.stats.iqr(X,0)/1.348
S_X = np.min([X_std, X_iqr], 0)

k = len(X)
Sigma = np.zeros([k, 2])
m = np.zeros(k)
for i in range(k):
    if i % 5000 == 0:
        print(i)
    mm = np.linalg.norm(Xdec-Xdec[i,:], axis = 1)
    mm = mm < k**(-1/6)
    m[i] = np.sum(mm)
    Sigma[i] = (m[i]/0.95*Rad**2*k**(1/3))**(-1/6)*S_X
print(np.mean(Sigma))
print(np.mean(m))    
#np.save('Sigma_JL', Sigma)