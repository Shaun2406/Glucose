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

def bivar_norm(x, y, idx):
    #Bivariate Normal Distribution for Ortho-Normalised Case (Covariance Matrix is Identity Matrix)
    pdf = np.zeros([len(x), len(y)])
    for i in range(len(x)):
        for j in range(len(y)):
            xy = np.matmul([x[i]-Xinmean[0], y[j]-Xinmean[1]], Ain)
            pdf[j,i] = 1/(2*np.pi*Sigma[idx]**2)*np.exp(-0.5*((xy[0]-Xindec[idx,0])**2+(xy[1]-Xindec[idx,1])**2)/Sigma[idx,0]**2)*detAin
    return pdf

def trivar_norm(x, y, z, idx):
    #Trivariate Normal Distribution for Ortho-Normalised Case (Covariance Matrix is Identity Matrix)
    pdf = np.zeros([len(x), len(y), len(z)])
    for i in range(len(x)):
        for j in range(len(y)):
            for k in range(len(z)):
                xyz = np.matmul([x[i]-Xmean[0], y[j]-Xmean[1], z[k]-Xmean[2]], A)
                pdf[j,i,k] = 1/((2*np.pi)**1.5*Sigma[idx]**3)*np.exp(-0.5*((xyz[0]-X[idx,0])**2+(xyz[1]-X[idx,1])**2+(xyz[2]-X[idx,2])**2)/Sigma[idx,0]**2)*detA
    return pdf

#LOADING DATA
GlucData = pd.read_csv('C:\WinPython-64bit-3.5.4.1Qt5\Glucose\GlucDataOverall.csv')
GlucData = GlucData.drop(['Unnamed: 0', 'Unnamed: 0.1', 'Operative', 'Patient', 't0', 'GF'], axis = 1)
GlucData['Gender'] = GlucData['Gender'] == 'female'
GlucData['Gender'] = GlucData['Gender'].astype(int)
features = ['SIt', 'Gt']
target = ['SIt+1']

#LOGGING RELEVANT DATA
GlucData = GlucData[GlucData['SIt'] > 0]
GlucData = GlucData[GlucData['SIt+1'] > 0]

GlucData['Gt'] = np.log10(GlucData['Gt'])
GlucData['SIt+1'] = np.log10(GlucData['SIt+1'])
GlucData['SIt'] = np.log10(GlucData['SIt'])

GlucData = GlucData.reset_index()

#Create an Ortho-Normalised Matrix Xdec - 3D
X = GlucData.loc[:, ['SIt', 'Gt', 'SIt+1']].values
C = np.cov(np.transpose(X))
R = np.linalg.cholesky(C)
A = np.linalg.inv(np.transpose(R))
detA = np.linalg.det(A)
Xmean = np.mean(X, 0)
X0 = X - Xmean
Xdec = np.matmul(X0, A)

#Create an Ortho-Normalised Matrix Xindec - 2D
Xin = GlucData.loc[:,['SIt', 'Gt']].values
Cin = np.cov(np.transpose(Xin))
Rin = np.linalg.cholesky(Cin)
Ain = np.linalg.inv(np.transpose(Rin))
detAin = np.linalg.det(Ain)
Xinmean = np.mean(Xin, 0)
Xin0 = Xin - Xinmean
Xindec = np.matmul(Xin0, Ain)

#Scaling Factors from Root Matrix (X), Standard Deviation and Max Range
Radii = np.sort(np.linalg.norm(Xdec, axis = 1))
R_X = Radii[len(Radii),:]
R_2X = Radii[np.round(len(Radii/2)),:]
Xstd = np.std(X,0)
Xiqr = scipy.stats.iqr(X,0)/1.348
S_X = np.min([Xstd, Xiqr],0)
k = len(Xdec)
M = np.zeros([k,3])
MB = np.zeros([k,1])
for i in range(k):
    if i % 1000 == 0:
        print(i)
    m = np.linalg.norm(Xdec-Xdec[i,:], axis = 1)
    m = m < k**(-1/6)
    M[i] = (np.sum(m)*R_X**3*k**(1/2))**(-1/6)
    MB[i] = (np.sum(m)*R_X**2*k**(1/3))**(-1/6)
    
Sigma = pd.DataFrame({'SigASIt': M[:,0], 'SigBGt': M[:,1], 'SigCSIt+1': M[:,2]})
Sigma.to_csv('C:\WinPython-64bit-3.5.4.1Qt5\Glucose\KernelSigma.csv')
'''Sigma = pd.read_csv('C:\WinPython-64bit-3.5.4.1Qt5\Glucose\KernelSigma.csv')'''
Sigma = Sigma.drop(['Unnamed: 0'], axis = 1)
Sigma = np.array(Sigma)

#Calculates Probability Field
Resolution = 150

SItx = np.linspace(-8.5, -1.5, Resolution)
Gtx = np.linspace(0.2, 1.4, Resolution)
SIt1x = np.linspace(-8.5, -1.5, Resolution) 
PDF = np.zeros([Resolution, Resolution])

for i in range(len(X)):
    PDF = PDF + bivar_norm(SItx, Gtx, i)
    if i % 1000 == 0:
        print(i)
np.save('C:\WinPython-64bit-3.5.4.1Qt5\Glucose\W2X.npy', PDF)
#PDF = np.loadtxt('C:\WinPython-64bit-3.5.4.1Qt5\Glucose\WXtotal.txt', delimiter = ',')

plt.figure()
plt.contour(SItx, Gtx, np.log(PDF), 100)
print(np.sum(PDF)*8.4/Resolution**2)
plt.xlabel('Sensitivity SIt')
plt.ylabel('Glucose Gt')