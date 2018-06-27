# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 12:06:41 2018

@author: smd118
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats
plt.close("all")

def Trap2D(Arr):
    l = len(Arr)-1
    corners = Arr[0,0]+Arr[l,l]+Arr[0,l]+Arr[l,0]
    edges = np.sum(Arr[1:l,0])+np.sum(Arr[1:l,l])+np.sum(Arr[0,1:l])+np.sum(Arr[l,1:l])
    middle = np.sum(Arr[1:l,1:l])
    tot = (middle*4+edges*2+corners)/4*8.4/(res-1)**2
    return tot

def Trap3D(Arr):
    l = len(Arr)-1
    corners = Arr[0,0,0]+Arr[0,0,l]+Arr[0,l,0]+Arr[l,0,0]+Arr[0,l,l]+Arr[l,0,l]+Arr[l,l,0]+Arr[l,l,l]
    edges1 = np.sum(Arr[1:l,0,0])+np.sum(Arr[1:l,0,l])+np.sum(Arr[1:l,l,0])+np.sum(Arr[1:l,l,l])
    edges2 = np.sum(Arr[0,1:l,0])+np.sum(Arr[0,1:l,l])+np.sum(Arr[l,1:l,0])+np.sum(Arr[l,1:l,l])
    edges3 = np.sum(Arr[0,0,1:l])+np.sum(Arr[0,l,1:l])+np.sum(Arr[l,0,1:l])+np.sum(Arr[l,l,1:l])
    edges = edges1+edges2+edges3
    faces = np.sum(Arr[0,1:l,1:l])+np.sum(Arr[1:l,0,1:l])+np.sum(Arr[1:l,1:l,0])+np.sum(Arr[l,1:l,1:l])+np.sum(Arr[1:l,l,1:l])+np.sum(Arr[1:l,1:l,l])
    middle = np.sum(Arr[1:l,1:l,1:l])
    tot = (middle*8+faces*4+edges*2+corners)/8*scale/(res-1)**3
    return tot

res = 150
def load_data():
    #LOADING DATA
    glucData = pd.read_csv('GlucDataOverall.csv')
    glucData = glucData.drop(['Unnamed: 0', 'Unnamed: 0.1', 'Operative', 'Patient', 't0', 'GF'], axis = 1)
    glucData['Gender'] = glucData['Gender'] == 'female'
    glucData['Gender'] = glucData['Gender'].astype(int)
    
    #LOGGING RELEVANT DATA
    glucData['Gt'] = np.log10(glucData['Gt'])
    glucData = glucData[np.isnan(glucData['Gt']) == 0]
    
    glucData['SIt+1'] = np.log10(glucData['SIt+1'])
    glucData = glucData[np.isnan(glucData['SIt+1']) == 0]
    
    glucData['SIt'] = np.log10(glucData['SIt'])
    glucData = glucData[np.isnan(glucData['SIt']) == 0]
    
    glucData = glucData.reset_index()
    return glucData

def transform(glucData):
    '''Create an Ortho-Normalised Matrix Xdec - 2D, for w(x)'''
    X = glucData.loc[:,['SIt', 'Gt', 'SIt+1']].values
    C = np.cov(np.transpose(X))
    R = np.linalg.cholesky(C)
    A = np.linalg.inv(np.transpose(R))
    detA = np.linalg.det(A)
    X0 = X - np.mean(X, 0)
    Xdec = np.matmul(X0, A)
    
    return (X, Xdec, detA, A)

glucData = load_data()
measured, measured_trf, trf_det, trf = transform(glucData)
    
grid_pts = [np.linspace(-8.5, -1.5, res), np.linspace(0.2, 1.4, res), np.linspace(-8.5, -1.5, res)]

W3a = np.load('C:\WinPython-64bit-3.5.4.1Qt5\Glucose\W3Xa.npy')
W3b = np.load('C:\WinPython-64bit-3.5.4.1Qt5\Glucose\W3Xb.npy')

W2D = np.loadtxt('C:\WinPython-64bit-3.5.4.1Qt5\Glucose\WXtotal.txt', delimiter = ',')
W3D = W3a + W3b
W2X = np.load('W2X.npy')
'''W3X = np.ones([150,150,150])
scale = (np.max(measured[:,0])-np.min(measured[:,0]))*(np.max(measured[:,1])-np.min(measured[:,1]))*(np.max(measured[:,2])-np.min(measured[:,2]))
print(Trap3D(W3b))'''
'''print(Trap2D(W2X))'''

fig = plt.figure()
grid_2D = np.meshgrid(grid_pts[0], grid_pts[1])
ax = fig.add_subplot(111, projection = '3d')
ax.plot_surface(grid_2D[0], grid_2D[1], W2X)
ax.set_xlabel('Sensitivity, SI(t)')
ax.set_ylabel('Glucose, G(t)')

plt.figure()
plt.contour(grid_2D[0], grid_2D[1], np.log(W2X), 100)

'''plt.figure()
plt.plot(grid_pts[2],W3D[105,105,:])   

print(np.sum(W3D)/res**3*(np.max(measured[:,0])-np.min(measured[:,0]))*(np.max(measured[:,1])-np.min(measured[:,1]))*(np.max(measured[:,2])-np.min(measured[:,2])))
print(np.sum(W2D)/res**2*(np.max(measured[:,0])-np.min(measured[:,0]))*(np.max(measured[:,1])-np.min(measured[:,1])))

for i in range(150):
    for j in range(150):
        if W2D[j,i] == 0:
            W2D[j,i] = 1            

for i in range(150):
    W3D[:,:,i] = W3D[:,:,i]/W2D
W3D = np.nan_to_num(W3D)    
print(np.sum(W3D)/res**3*(np.max(measured[:,0])-np.min(measured[:,0]))*(np.max(measured[:,1])-np.min(measured[:,1]))*(np.max(measured[:,2])-np.min(measured[:,2]))) 

plt.figure()
plt.contour(grid_pts[0], grid_pts[1], np.log(W3D[:,:,75]))   

plt.figure()
plt.plot(grid_pts[2],W3D[105,105,:])   '''