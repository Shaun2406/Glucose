# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 12:06:41 2018

Non-parallel calculation of probability field, as well as calculation of sigma matrix
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.close("all")

def bivar_norm(x, y, idx):
    #Bivariate Normal Distribution for Ortho-Normalised Case (Covariance Matrix is Identity Matrix)
    pdf = np.zeros([len(x), len(y)])
    for i in range(len(x)):
        for j in range(len(y)):
            xy = np.matmul([x[i]-Xinmean[0], y[j]-Xinmean[1]], Ain)
            pdf[j,i] = 1/(2*np.pi*Sigma[idx]**2)*np.exp(-0.5*((xy[0]-Xindec[idx,0])**2+(xy[1]-Xindec[idx,1])**2)/Sigma[idx]**2)*detAin
    return pdf

def trivar_norm(x, y, z, idx):
    #Trivariate Normal Distribution for Ortho-Normalised Case (Covariance Matrix is Identity Matrix)
    pdf = np.zeros([len(x), len(y), len(z)])
    for i in range(len(x)):
        for j in range(len(y)):
            for k in range(len(z)):
                xyz = np.matmul([x[i]-Xmean[0], y[j]-Xmean[1], z[k]-Xmean[2]], A)
                pdf[j,i,k] = 1/((2*np.pi)**1.5*Sigma[idx]**3)*np.exp(-0.5*((xyz[0]-X[idx,0])**2+(xyz[1]-X[idx,1])**2+(xyz[2]-X[idx,2])**2)/Sigma[idx]**2)*detA
    return pdf

def Trap2D(Arr):
    l = len(Arr)-1
    corners = Arr[0,0]+Arr[l,l]+Arr[0,l]+Arr[l,0]
    edges = np.sum(Arr[1:l,0])+np.sum(Arr[1:l,l])+np.sum(Arr[0,1:l])+np.sum(Arr[l,1:l])
    middle = np.sum(Arr[1:l,1:l])
    tot = (middle*4+edges*2+corners)/4*8.4/(Resolution-1)**2
    return tot

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

'''#Scaling Factors from Root Matrix (X), Standard Deviation and Max Range - 3D
Rad = np.sort(np.linalg.norm(Xdec, axis = 1))
R_X = Rad[len(Rad)-1]
R_2X = Rad[round(len(Rad)*0.95)]
X_s = np.std(X,0)
k = len(X)
m = np.zeros([k])
M = np.zeros([k])
MB = np.zeros([k])
MC = np.zeros([k])
for i in range(k):
    if i % 1000 == 0:
        print(i)
    mm = np.linalg.norm(Xdec-Xdec[i,:], axis = 1)
    mm = mm < k**(-1/6)
    m[i] = np.sum(mm)
    M[i] = (m[i]*R_X**3*k**(1/2))**(-1/6)
    MB[i] = (m[i]*R_X**2*k**(1/3))**(-1/6)
    MC[i] = (m[i]/0.95*R_2X**3*k**(1/2))**(-1/6)

#Scaling Factors from Root Matrix (X), Standard Deviation and Max Range - 2D
Radin = np.sort(np.linalg.norm(Xindec, axis = 1))
R_Xin = Radin[len(Radin)-1]
X_sin = np.std(Xin,0)
k = len(Xin)
m_in = np.zeros([k])
M_in = np.zeros([k])
for i in range(k):
    if i % 1000 == 0:
        print(i)
    mm_in = np.linalg.norm(Xindec-Xindec[i,:], axis = 1)
    mm_in = mm_in < k**(-1/6)
    m_in[i] = np.sum(mm_in)
    M_in[i] = (m_in[i]*R_Xin**2*k**(1/3))**(-1/6)'''
    
#np.save('Sigma', M)
Sigma = np.load('Sigma_3D.npy')

#Calculates Probability Field
Resolution = 150

SItx = np.linspace(-8.5, -1.5, Resolution)
Gtx = np.linspace(0.2, 1.4, Resolution)
SIt1x = np.linspace(-8.5, -1.5, Resolution) 
PDF = np.zeros([Resolution, Resolution])
print(np.mean(10**GlucData['SIt+1']))
'''for i in range(len(X)):
    PDF = PDF + bivar_norm(SItx, Gtx, i)
    if i == 1000:
        print(i)
np.save('C:\WinPython-64bit-3.5.4.1Qt5\Glucose\W2X.npy', PDF)
#PDF = np.loadtxt('C:\WinPython-64bit-3.5.4.1Qt5\Glucose\WXtotal.txt', delimiter = ',')

plt.figure()
plt.contour(SItx, Gtx, PDF, 100)
print(Trap2D(PDF))
plt.xlabel('Sensitivity SIt')
plt.ylabel('Glucose Gt')'''