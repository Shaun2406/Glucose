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

NUM_THREADS = 8

def init_worker(lock_, out_, means_, grid_pts_, trf_, res_):
    global lock, out, means, grid_pts, trf, res
    lock = lock_
    out = out_
    means = means_
    grid_pts = grid_pts_
    trf = trf_
    res = res_

def trivar_norm(measured_pt, sigma):
    density_func = np.zeros([len(grid_pts[0]), len(grid_pts[1]), len(grid_pts[2])])
    for x in range(len(grid_pts[0])):
        for y in range(len(grid_pts[1])):
            for z in range(len(grid_pts[2])):
                xyz = np.matmul([grid_pts[0][x]-means[0], grid_pts[1][y]-means[1], grid_pts[2][z]-means[2]], trf)
                density_func[y,x,z] = 1/(2*np.pi*sigma**3)**(3/2)*np.exp(-1/2*((xyz[0]-measured_pt[0])**2/sigma**2+(xyz[1]-measured_pt[1])**2/sigma**2+(xyz[2]-measured_pt[2])**2/sigma**2))
    return density_func
    '''out_array = np.frombuffer(out.get_obj()).reshape((res, res, res))
    lock.acquire()
    try:
        out_array += density_func
    finally:
        lock.release()'''

'''LOADING DATA'''
GlucData = pd.read_csv('C:\WinPython-64bit-3.5.4.1Qt5\Glucose\GlucDataOverall.csv')
GlucData = GlucData.drop(['Unnamed: 0', 'Unnamed: 0.1', 'Operative', 'Patient', 't0', 'GF'], axis = 1)
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


'''Create an Ortho-Normalised Matrix Xdec - 3D'''
X = GlucData.loc[:, ['SIt', 'Gt', 'SIt+1']].values

X = X[1:10,:]
C = np.cov(np.transpose(X))
R = np.linalg.cholesky(C)
trf = np.linalg.inv(np.transpose(R))
X0 = X - np.mean(X, 0)
measured_trf = np.matmul(X0, trf)

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
sigma = pd.read_csv('C:\WinPython-64bit-3.5.4.1Qt5\Glucose\KernelSigma.csv')
sigma = sigma.drop(['Unnamed: 0'], axis = 1)
sigma = np.array(sigma)

if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    res = 2
    grid_pts = [np.linspace(np.min(X[:,0]), np.max(X[:,0]), res),
                np.linspace(np.min(X[:,1]), np.max(X[:,1]), res),
                np.linspace(np.min(X[:,2]), np.max(X[:,2]), res)]
PDF = np.zeros([res, res, res])
means = np.mean(X, 0)

density_func_raw = Array('d', res**3)
density_func = np.frombuffer(density_func_raw.get_obj()).reshape((res, res, res))
density_func.fill(0)
start = time()
with Pool(processes=8, initializer=init_worker, initargs=(Lock(), density_func_raw, means, grid_pts, trf, res)) as pool:
    pool.starmap(trivar_norm, [(measured_trf[i], sigma[i][0]) for i in range(len(measured_trf))])
print(time() - start)
print(density_func)

'''start = time()
for i in range(len(X)):
    PDF = PDF + trivar_norm(measured_trf[i], sigma[i,0])
    if i % 1000 == 0:
        print(i)
print(time() - start)
np.save('C:\WinPython-64bit-3.5.4.1Qt5\Glucose\W3X', PDF)
PDF = np.load('C:\WinPython-64bit-3.5.4.1Qt5\Glucose\W3X.npy')

print(np.sum(PDF)*(np.max(X[:,0])-np.min(X[:,0]))/Resolution*(np.max(X[:,1])-np.min(X[:,1]))/Resolution*(np.max(X[:,2])-np.min(X[:,2]))/Resolution)
plt.figure()
plt.contour(SItx, Gtx, np.log(PDF[:,:,100]), 100)
plt.figure()
plt.contour(SItx, SIt1x, np.log(PDF[:,100,:]), 100)
plt.figure()
plt.contour(Gtx, SIt1x, np.log(PDF[100,:,:]), 100)'''
