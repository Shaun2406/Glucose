# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 12:06:41 2018

@author: smd118
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    
def centre_pts(grid_pts, measured_trf, means):
    #Bivariate Normal Distribution for Ortho-Normalised Case (Covariance Matrix is Identity Matrix)
    x_out = np.zeros([len(grid_pts[0]), len(grid_pts[1])])
    y_out = np.zeros([len(grid_pts[0]), len(grid_pts[1])])
    X_pts = np.zeros(len(measured_trf))
    Y_pts = np.zeros([len(measured_trf), len(grid_pts[1])])
    for x in range(len(grid_pts[0])):
        for y in range(len(grid_pts[1])):
            xy = np.matmul([grid_pts[0][x]-means[0], grid_pts[1][y]-means[1]], trf)
            x_out[y,x] = xy[0]
            y_out[y,x] = xy[1]
    for i in range(len(measured_trf)):
        X_pts[i] = np.argmin(abs(x_out[0,:]-measured_trf[i,0]))
        Y_pts[i,:] = np.argmin(abs(y_out-measured_trf[i,1]),0)
    return X_pts, Y_pts

def bivar_norm(measured_pt, sigma, X_pts, Y_pts):
    density_func = np.zeros([len(grid_pts[0]), len(grid_pts[1])])
    for x in range(int(np.max([X_pts-18,0])),int(np.min([X_pts+18,150]))):
        for y in range(int(np.max([Y_pts[x]-18,0])),int(np.min([Y_pts[x]+18,150]))):
            xy = np.matmul([grid_pts[0][x]-means[0], grid_pts[1][y]-means[1]], trf)
            density_func[y,x] = 1/(2*np.pi*sigma**2)*np.exp(-0.5*((xy[0]-measured_pt[0])**2+(xy[1]-measured_pt[1])**2)/sigma**2)
    out_array = np.frombuffer(out.get_obj()).reshape((res, res))
    lock.acquire()
    try:
        out_array += density_func
    finally:
        lock.release()

def Trap2D(Arr):
    l = len(Arr)-1
    corners = Arr[0,0]+Arr[l,l]+Arr[0,l]+Arr[l,0]
    edges = np.sum(Arr[1:l,0])+np.sum(Arr[1:l,l])+np.sum(Arr[0,1:l])+np.sum(Arr[l,1:l])
    middle = np.sum(Arr[1:l,1:l])
    tot = (middle*4+edges*2+corners)/4*8.4/(res-1)**2
    return tot

def load_data():
    #Loading and transforming data
    GlucData = pd.read_csv('GlucDataOverall.csv')
    GlucData = GlucData[GlucData['SIt'] > 0]
    GlucData = GlucData[GlucData['SIt+1'] > 0]
    
    GlucData['Gt'] = np.log10(GlucData['Gt'])
    GlucData['SIt+1'] = np.log10(GlucData['SIt+1'])
    GlucData['SIt'] = np.log10(GlucData['SIt'])
    
    GlucData = GlucData.reset_index()
    return GlucData

def transform(glucData):
    #Create an Ortho-Normalised Matrix Xdec - 2D
    Xin = glucData.loc[:,['SIt', 'Gt']].values
    Cin = np.cov(np.transpose(Xin))
    Rin = np.linalg.cholesky(Cin)
    Ain = np.linalg.inv(np.transpose(Rin))
    detAin = np.linalg.det(Ain)
    Xin0 = Xin - np.mean(Xin, 0)
    Xindec = np.matmul(Xin0, Ain)
    
    #Xindec = Xindec[0:10,:]
    
    return (Xin, Xindec, detAin, Ain)

if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    glucData = load_data()
    measured, measured_trf, trf_det, trf = transform(glucData)

    sigma = np.load('Sigma.npy')   
    res = 150
    
    grid_pts = [np.linspace(-8.5, -1.5, res), np.linspace(0.2, 1.4, res)]
    means = np.mean(measured, 0)
    X_pts, Y_pts = centre_pts(grid_pts, measured_trf, means)

    density_func_raw = Array('d', res**2)
    density_func = np.frombuffer(density_func_raw.get_obj()).reshape((res, res))
    density_func.fill(0)
    start = time()
    with Pool(processes=8, initializer=init_worker, initargs=(Lock(), density_func_raw, means, grid_pts, trf, res)) as pool:
        pool.starmap(bivar_norm, [(measured_trf[i], sigma[i], X_pts[i], Y_pts[i,:]) for i in range(len(measured_trf))])
    print(time() - start)
    density_func = density_func*trf_det
    np.save('C:\WinPython-64bit-3.5.4.1Qt5\Glucose\W2X', density_func)
    print(Trap2D(density_func))