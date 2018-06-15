# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 12:06:41 2018

@author: smd118
"""

import sklearn as skl
import sklearn.preprocessing as sklpp
import sklearn.decomposition as skldc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model
from sklearn import svm
from sklearn import gaussian_process
import scipy
plt.close("all")


GlucData = pd.read_csv('C:\WinPython-64bit-3.5.4.1Qt5\Glucose\GlucDataOverall.csv')
GlucData = GlucData.drop(['Unnamed: 0', 'Unnamed: 0.1', 'Operative', 'Patient', 't0', 'GF'], axis = 1)
GlucData['Gender'] = GlucData['Gender'] == 'female'
GlucData['Gender'] = GlucData['Gender'].astype(int)
features = ['Gt', 'Gt-1', 'Gt-2', 'Pt', 'Pt-1', 'Pt-2', 'SIt', 'SIt-1', 'SIt-2', 'ut', 'ut-1', 'ut-2']
features = ['SIt', 'Gt']
target = ['SIt+1']

GlucData['Gt'] = np.log10(GlucData['Gt'])
GlucData = GlucData[np.isnan(GlucData['Gt']) == 0]

GlucData['SIt+1'] = np.log10(GlucData['SIt+1'])
GlucData = GlucData[np.isnan(GlucData['SIt+1']) == 0]

GlucData['SIt'] = np.log10(GlucData['SIt'])
GlucData = GlucData[np.isnan(GlucData['SIt']) == 0]

'''GlucData['SIt-1'] = np.log10(GlucData['SIt-1'])
GlucData = GlucData[np.isnan(GlucData['SIt-1']) == 0]'''

GlucData = GlucData.reset_index()

x = GlucData.loc[:, features].values
y = GlucData.loc[:, target].values
y = sklpp.StandardScaler().fit_transform(y)
x = sklpp.StandardScaler().fit_transform(x)

'''pca = skldc.PCA(n_components = 3)

princomps = pca.fit_transform(x)
principalDF = pd.DataFrame(data = princomps, columns = ['princomp1', 'princomp2', 'princomp3'])
finalDF = pd.concat([principalDF, GlucData['SIt+1']], axis = 1)
Minim = np.min(GlucData['SIt+1'])
Maxim = np.max(GlucData['SIt+1'])
GlucCol = (GlucData['SIt+1']-Minim)/(Maxim-Minim)

AA = pca.components_
BB = pca.explained_variance_ratio_'''

'''fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
for i in range(len(GlucData['SIt+1'])):
    if i % 20 == 0:
        plt.scatter(finalDF.loc[i, 'princomp1'], finalDF.loc[i, 'princomp2'], color = [GlucCol[i], GlucCol[i], GlucCol[i]])'''

pd.DataFrame.hist(GlucData, 'SIt+1', bins = 150)

lm = linear_model.LinearRegression()
model = lm.fit(x,y)
print(lm.score(x,y))

'''plt.figure()
plt.plot(GlucData['SIt'], GlucData['SIt+1'], 'kx')

plt.figure()
plt.plot(GlucData['Gt'], GlucData['SIt+1'], 'kx')

plt.figure()
plt.plot(GlucData['SIt-1'], GlucData['SIt+1'], 'kx')'''

'''QQ = np.sort(abs(lm.coef_))
A = np.zeros(11)
for i in range(len(QQ[0,:])-1):
    A[i] = QQ[0,i+1]/QQ[0,i]
    
print(A)
print(lm.coef_)
print(np.sort(abs(lm.coef_)))'''

'''GlucData['Gt'] = np.log10(GlucData['Gt'])
GlucData = GlucData[np.isnan(GlucData['Gt']) == 0]'''

pd.DataFrame.hist(GlucData, 'Gt', bins = 30)

'''fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.plot(GlucData['SIt'], GlucData['Gt'], GlucData['SIt+1'], 'kx')
plt.xlabel('Sensitivity SIt')
plt.ylabel('Glucose Gt')'''

'''plt.figure()
plt.plot(GlucData['t'], GlucData['SIt+1'], 'kx')
plt.title('Sensitivity vs Time Since Protocol Start')
plt.xlabel('Time, t (mins)')
plt.ylabel('Log Sensitivity, SI')
plt.xlim([0, 60000])'''

X = GlucData.loc[:, ['SIt', 'Gt', 'SIt+1']].values
C = np.cov(np.transpose(X))
R = np.linalg.cholesky(C)
A = np.linalg.inv(np.transpose(R))

XX = X - np.mean(X, 0)
Xdec = np.matmul(XX, A)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.plot(Xdec[:,0], Xdec[:,1], Xdec[:,2],'kx')
plt.xlabel('Sensitivity SIt')
plt.ylabel('Glucose Gt')

Rdec = np.max(np.sqrt(Xdec[:,0]**2+Xdec[:,1]**2+Xdec[:,2]**2))
Q = np.std(X,0)
QQ = scipy.stats.iqr(X,0)/1.348
XXX = np.min([Q, QQ],0)