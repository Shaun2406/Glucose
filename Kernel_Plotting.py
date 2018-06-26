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
from time import time
from multiprocessing import Pool, Lock, Array
plt.close("all")

def trivar_norm(measured_pt, sigma):
    density_func = np.zeros([len(grid_pts[0]), len(grid_pts[1]), len(grid_pts[2])])
    for x in range(len(grid_pts[0])):
        for y in range(len(grid_pts[1])):
            for z in range(len(grid_pts[2])):
                xyz = np.matmul([grid_pts[0][x]-means[0], grid_pts[1][y]-means[1], grid_pts[2][z]-means[2]], trf)
                density_func[y,x,z] = 1/((2*np.pi)**(3/2)*sigma**3)*np.exp(-1/2*((xyz[0]-measured_pt[0])**2/sigma**2+(xyz[1]-measured_pt[1])**2/sigma**2+(xyz[2]-measured_pt[2])**2/sigma**2))
    out_array = np.frombuffer(out.get_obj()).reshape((res, res, res))
    lock.acquire()
    try:
        out_array += density_func
    finally:
        lock.release()

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
    Xin = glucData.loc[:,['SIt', 'Gt', 'SIt+1']].values
    Xin = Xin[0:37554,:]
    Cin = np.cov(np.transpose(Xin))
    Rin = np.linalg.cholesky(Cin)
    Ain = np.linalg.inv(np.transpose(Rin))
    detAin = np.linalg.det(Ain)
    Xin0 = Xin - np.mean(Xin, 0)
    Xindec = np.matmul(Xin0, Ain)
    
    return (Xin, Xindec, detAin, Ain)

glucData = load_data()
measured, measured_trf, trf_det, trf = transform(glucData)
    
sigma = pd.read_csv('KernelSigma.csv')
sigma = sigma.drop(['Unnamed: 0'], axis = 1)
sigma = np.array(sigma)
    
res = 150
    
grid_pts = [np.linspace(np.min(measured[:,0]), np.max(measured[:,0]), res),
            np.linspace(np.min(measured[:,1]), np.max(measured[:,1]), res),
            np.linspace(np.min(measured[:,2]), np.max(measured[:,2]), res)]
means = np.mean(measured, 0)

W3a = np.load('C:\WinPython-64bit-3.5.4.1Qt5\Glucose\W3Xa.npy')
W3b = np.load('C:\WinPython-64bit-3.5.4.1Qt5\Glucose\W3Xb.npy')

W2D = np.loadtxt('C:\WinPython-64bit-3.5.4.1Qt5\Glucose\WXtotal.txt', delimiter = ',')
W3D = W3a + W3b

plt.figure()
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
plt.plot(grid_pts[2],W3D[105,105,:])   