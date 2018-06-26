# -*- coding: utf-8 -*-
"""
Exploration of data including multiple linear regression, principal components and various histograms
"""

import sklearn as skl
import sklearn.preprocessing as sklpp
import sklearn.decomposition as skldc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model
plt.close("all")


GlucData = pd.read_csv('C:\WinPython-64bit-3.5.4.1Qt5\Glucose\GlucDataOverall.csv')
GlucData = GlucData.drop(['Unnamed: 0', 'Unnamed: 0.1', 'Operative', 'Patient', 't0', 'GF'], axis = 1)
GlucData['Gender'] = GlucData['Gender'] == 'female'
GlucData['Gender'] = GlucData['Gender'].astype(int)
features = ['Gt', 'Gt-1', 'Gt-2', 'Pt', 'Pt-1', 'Pt-2', 'SIt', 'SIt-1', 'SIt-2', 'ut', 'ut-1', 'ut-2']
features = ['SIt', 'Gt']
target = ['SIt+1']

GlucData['SIt+1'] = np.log10(GlucData['SIt+1'])
GlucData = GlucData[np.isnan(GlucData['SIt+1']) == 0]

GlucData['SIt'] = np.log10(GlucData['SIt'])
GlucData = GlucData[np.isnan(GlucData['SIt']) == 0]

GlucData['SIt-1'] = np.log10(GlucData['SIt-1'])
GlucData = GlucData[np.isnan(GlucData['SIt-1']) == 0]

GlucData['SIt-2'] = np.log10(GlucData['SIt-2'])
GlucData = GlucData[np.isnan(GlucData['SIt-2']) == 0]
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

plt.figure()
plt.plot(GlucData['SIt'], GlucData['SIt+1'], 'kx')

plt.figure()
plt.plot(GlucData['Gt'], GlucData['SIt+1'], 'kx')

plt.figure()
plt.plot(GlucData['SIt-1'], GlucData['SIt+1'], 'kx')

'''QQ = np.sort(abs(lm.coef_))
A = np.zeros(11)
for i in range(len(QQ[0,:])-1):
    A[i] = QQ[0,i+1]/QQ[0,i]
    
print(A)
print(lm.coef_)
print(np.sort(abs(lm.coef_)))'''

GlucData['Gt'] = np.log10(GlucData['Gt'])
GlucData = GlucData[np.isnan(GlucData['Gt']) == 0]

pd.DataFrame.hist(GlucData, 'Gt', bins = 30)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.plot(GlucData['SIt'], GlucData['Gt'], GlucData['SIt+1'], 'kx')
plt.xlabel('Sensitivity SIt')
plt.ylabel('Glucose Gt')
    