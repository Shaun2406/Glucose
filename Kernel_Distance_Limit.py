# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 12:06:41 2018

Non-parallel calculation of probability field, as well as calculation of sigma matrix
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
plt.close("all")

def bivar_norm(x, y, idx):
    #Bivariate Normal Distribution for Ortho-Normalised Case (Covariance Matrix is Identity Matrix)
    pdf = np.zeros([len(x), len(y)])
    x_out = np.zeros([len(x), len(y)])
    y_out = np.zeros([len(x), len(y)])
    for i in range(len(x)):
        for j in range(len(y)):
            xy = np.matmul([x[i]-Xinmean[0], y[j]-Xinmean[1]], Ain)
            pdf[j,i] = 1/(2*np.pi*Sigma[idx]**2)*np.exp(-0.5*((xy[0]-Xindec[idx,0])**2+(xy[1]-Xindec[idx,1])**2)/Sigma[idx]**2)*detAin
            x_out[j,i] = xy[0]
            y_out[j,i] = xy[1]
    return pdf, x_out, y_out

def bivar_norm_eff(x, y, idx):
    #Bivariate Normal Distribution for Ortho-Normalised Case (Covariance Matrix is Identity Matrix)
    X_pts = np.argmin(abs(x_out[0,:]-Xindec[idx,0]))
    Y_pts = np.argmin(abs(y_out-Xindec[idx,1]),0)
    pdf = np.zeros([len(x), len(y)])
    for i in range(np.max([X_pts-17,0]),np.min([X_pts+17,150])):
        for j in range(np.max([Y_pts[i]-17,0]),np.min([Y_pts[i]+17,150])):
            xy = np.matmul([x[i]-Xinmean[0], y[j]-Xinmean[1]], Ain)
            pdf[j,i] = 1/(2*np.pi*Sigma[idx]**2)*np.exp(-0.5*((xy[0]-Xindec[idx,0])**2+(xy[1]-Xindec[idx,1])**2)/Sigma[idx]**2)*detAin
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

#Calculates Probability Field
Resolution = 150
Sigma = np.load('Sigma.npy')

SItx = np.linspace(-8.5, -1.5, Resolution)
Gtx = np.linspace(0.2, 1.4, Resolution)
SIt1x = np.linspace(-8.5, -1.5, Resolution) 
PDF = np.zeros([Resolution, Resolution])

i = 24
_, x_out, y_out = bivar_norm(SItx, Gtx, i)


start = time()
for i in range(100):
    PDF = PDF + bivar_norm_eff(SItx, Gtx, i)
'''np.save('C:\WinPython-64bit-3.5.4.1Qt5\Glucose\W2X.npy', PDF)'''
print(time()-start)
print(Trap2D(PDF))

'''loc = np.meshgrid(SItx, Gtx)

plt.figure()
plt.contour(loc[0], loc[1], PDF, 100)
plt.plot(Xin[i,0], Xin[i,1], 'kx')

plt.xlabel('Sensitivity SIt')
plt.ylabel('Glucose Gt')

zero_map = (PDF == 0)

plt.figure()
plt.contour(x_out, y_out, PDF, 100)
plt.plot(Xindec[i,0], Xindec[i,1], 'kx')
plt.xlabel('Sensitivity SIt')
plt.ylabel('Glucose Gt')
plt.axis('equal')

plt.figure()
plt.contourf(x_out, y_out, zero_map, 100)
plt.plot(Xindec[i,0], Xindec[i,1], 'kx')
plt.xlabel('Sensitivity SIt')
plt.ylabel('Glucose Gt')
plt.axis('equal')

idx = 24
pdf_1d = np.zeros(150)
yy = np.linspace(np.min(y_out), np.max(y_out), 150)
for j in range(150):
    pdf_1d[j] = 1/(2*np.pi*Sigma[idx]**2)*np.exp(-0.5*((yy[j]-Xindec[idx,1])**2/Sigma[idx]**2))*detAin
plt.figure()
plt.plot(yy, pdf_1d)


print(X_pts)
print(Y_pts)'''