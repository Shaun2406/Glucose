# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 12:06:41 2018

@author: smd118
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from multiprocessing import Pool
import scipy.stats
plt.close("all")

def bivar_norm(x, y):
    '''Bivariate Normal Distribution for Ortho-Normalised Case (Covariance Matrix is Identity Matrix)'''
    xy = np.matmul([x-Xinmean[0], y-Xinmean[1]], Ain)
    pdf = 1/(2*np.pi*Sigma[idx,0]*Sigma[idx,1])*np.exp(-1/2*((xy[0]-Xindec[idx,0])**2/Sigma[idx,0]**2+(xy[1]-Xindec[idx,1])**2/Sigma[idx,1]**2))*detAin
    return pdf

def data_stream(a, b):
    for i, av in enumerate(a):
        for j, bv in enumerate(b):
            yield (i, j), (av, bv)
            
def myfunc(args):
    return args[0], bivar_norm(*args[1])

def trivar_norm(x, y, z, idx):
    '''Trivariate Normal Distribution for Ortho-Normalised Case (Covariance Matrix is Identity Matrix)'''
    pdf = np.zeros([len(x), len(y), len(z)])
    for i in range(len(x)):
        for j in range(len(y)):
            for k in range(len(z)):
                xyz = np.matmul([x[i]-Xinmean[0], y[j]-Xinmean[1], z[k]-Xinmean[2]], A)
                pdf[i,j,k] = 1/(2*np.pi*Sigma[idx,0]*Sigma[idx,1]*Sigma[idx,2])**(3/2)*np.exp(-1/2*((xyz[0]-X[idx,0])**2/Sigma[idx,0]**2+(xyz[1]-X[idx,1])**2/Sigma[idx,1]**2+(xyz[2]-X[idx,2])**2/Sigma[idx,2]**2))
    return pdf

'''LOADING DATA'''
GlucData = pd.read_csv('C:\WinPython-64bit-3.5.4.1Qt5\Glucose\GlucDataOverall.csv')
GlucData = GlucData.drop(['Unnamed: 0', 'Unnamed: 0.1', 'Operative', 'Patient', 't0', 'GF'], axis = 1)
GlucData['Gender'] = GlucData['Gender'] == 'female'
GlucData['Gender'] = GlucData['Gender'].astype(int)
features = ['Gt', 'Gt-1', 'Gt-2', 'Pt', 'Pt-1', 'Pt-2', 'SIt', 'SIt-1', 'SIt-2', 'ut', 'ut-1', 'ut-2']
features = ['SIt', 'Gt']
target = ['SIt+1']

'''LOGGING RELEVANT DATA'''
GlucData['Gt'] = np.log10(GlucData['Gt'])
GlucData = GlucData[np.isnan(GlucData['Gt']) == 0]

GlucData['SIt+1'] = np.log10(GlucData['SIt+1'])
GlucData = GlucData[np.isnan(GlucData['SIt+1']) == 0]

GlucData['SIt'] = np.log10(GlucData['SIt'])
GlucData = GlucData[np.isnan(GlucData['SIt']) == 0]

GlucData = GlucData.reset_index()


'''HISTOGRAMS'''
'''pd.DataFrame.hist(GlucData, 'SIt+1', bins = 150)
pd.DataFrame.hist(GlucData, 'Gt', bins = 30)

plt.figure()
plt.plot(GlucData['SIt'], GlucData['SIt+1'], 'kx')

plt.figure()
plt.plot(GlucData['Gt'], GlucData['SIt+1'], 'kx')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.plot(GlucData['SIt'], GlucData['Gt'], GlucData['SIt+1'], 'kx')
plt.xlabel('Sensitivity SIt')
plt.ylabel('Glucose Gt')'''

'''Create an Ortho-Normalised Matrix Xdec - 3D'''
X = GlucData.loc[:, ['SIt', 'Gt', 'SIt+1']].values
C = np.cov(np.transpose(X))
R = np.linalg.cholesky(C)
A = np.linalg.inv(np.transpose(R))
X0 = X - np.mean(X, 0)
Xdec = np.matmul(X0, A)

'''Create an Ortho-Normalised Matrix Xdec - 2D, for w(x)'''
Xin = GlucData.loc[:,['SIt', 'Gt']].values
Cin = np.cov(np.transpose(Xin))
Rin = np.linalg.cholesky(Cin)
Ain = np.linalg.inv(np.transpose(Rin))
detAin = np.linalg.det(Ain)
Xin0 = Xin - np.mean(Xin, 0)
Xindec = np.matmul(Xin0, Ain)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.plot(Xdec[:,0], Xdec[:,1], Xdec[:,2],'kx')
plt.xlabel('Sensitivity SIt')
plt.ylabel('Glucose Gt')

'''Scaling Factors from Root Matrix (X), Standard Deviation and Max Range'''
'''R_X = np.max(np.sqrt(Xdec[:,0]**2+Xdec[:,1]**2+Xdec[:,2]**2))
Xstd = np.std(X,0)
Xiqr = scipy.stats.iqr(X,0)/1.348
S_X = np.min([Xstd, Xiqr],0)
k = len(Xdec)
M = np.zeros([k,3])
for i in range(k):
    if i % 1000 == 0:
        print(i)
    m = np.linalg.norm(Xdec-Xdec[i,:], axis = 1)
    m = m < k**(-1/6)
    M[i] = (np.sum(m)*R_X**3*k**(1/2))**(-1/6)
Sigma = pd.DataFrame({'SigASIt': M[:,0], 'SigBGt': M[:,1], 'SigCSIt+1': M[:,2]})
Sigma.to_csv('C:\WinPython-64bit-3.5.4.1Qt5\Glucose\KernelSigma.csv')'''
Sigma = pd.read_csv('C:\WinPython-64bit-3.5.4.1Qt5\Glucose\KernelSigma.csv')
Sigma = Sigma.drop(['Unnamed: 0'], axis = 1)
Sigma = np.array(Sigma)

Resolution = 150

SItx = np.linspace(np.min(Xin[:,0]), np.max(Xin[:,0]), Resolution)
Gtx = np.linspace(np.min(Xin[:,1]), np.max(Xin[:,1]), Resolution) 
PDF = np.zeros([Resolution, Resolution])
Xinmean = np.mean(Xin, 0)
for i in range(1000):
    if __name__ == "__main__":
        __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
        idx = i
        global di
        pool = Pool(processes=12)
        PDFi = pool.map(myfunc, data_stream(SItx, Gtx))
        print(PDFi)
        PDF = PDF + PDFi
        if i % 1000 == 0:
            print(i)
np.savetxt('C:\WinPython-64bit-3.5.4.1Qt5\Glucose\WX.txt', PDF, delimiter=',')
'''PDF = np.loadtxt('C:\WinPython-64bit-3.5.4.1Qt5\Glucose\WXtotal.txt', delimiter=',')'''

plt.figure()
plt.contour(SItx, Gtx, np.log(PDF), 100)
print(np.sum(PDF)*(np.max(X[:,0])-np.min(X[:,0]))/Resolution*(np.max(X[:,1])-np.min(X[:,1]))/Resolution)
plt.xlabel('Sensitivity SIt')
plt.ylabel('Glucose Gt')

plt.figure()
plt.plot(Xindec[:,0], Xindec[:,1], 'kx')

plt.figure()
plt.plot(Xin[:,0], Xin[:,1], 'kx')