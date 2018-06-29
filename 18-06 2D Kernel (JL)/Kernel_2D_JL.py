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

def init_worker(lock_, out_, grid_pts_, res_):
    global lock, out, grid_pts, res
    lock = lock_
    out = out_
    grid_pts = grid_pts_
    res = res_
    
def bivar_norm(measured_pt, sigma, X_pts, Y_pts):
    density_func = np.zeros([len(grid_pts[0]), len(grid_pts[1])])
    for x in range(int(np.max([X_pts-20,0])),int(np.min([X_pts+20,150]))):
        for y in range(int(np.max([Y_pts-20,0])),int(np.min([Y_pts+20,150]))):
            density_func[y,x] = 1/(2*np.pi*sigma[0]*sigma[1])*np.exp(-0.5*((grid_pts[0][x]-measured_pt[0])**2/sigma[0]**2+(grid_pts[1][y]-measured_pt[1])**2/sigma[1]**2))
    '''density_func = density_func/Trap2D(density_func)'''
    out_array = np.frombuffer(out.get_obj()).reshape((res, res))
    lock.acquire()
    try:
        out_array += density_func
    finally:
        lock.release()
        
def centre_pts(grid_pts, measured):
    #Bivariate Normal Distribution for Ortho-Normalised Case (Covariance Matrix is Identity Matrix)
    for i in range(len(measured)):
        X_pts = np.zeros(len(measured))
        Y_pts = np.zeros(len(measured))
        X_pts[i] = np.argmin(abs(grid_pts[0]-measured[i,0]))
        Y_pts[i] = np.argmin(abs(grid_pts[1]-measured[i,1]))
    return X_pts, Y_pts

def Trap2D(Arr):
    l = len(Arr)-1
    corners = Arr[0,0]+Arr[l,l]+Arr[0,l]+Arr[l,0]
    edges = np.sum(Arr[1:l,0])+np.sum(Arr[1:l,l])+np.sum(Arr[0,1:l])+np.sum(Arr[l,1:l])
    middle = np.sum(Arr[1:l,1:l])
    tot = (middle*4+edges*2+corners)/4*(10**-1.8-10**-8.5)**2/(res-1)**2
    return tot

def load_data():
    #Loading and transforming data
    GlucData = pd.read_csv('C:\\WinPython-64bit-3.5.4.1Qt5\\Glucose\\GlucDataOverall.csv')
    GlucData = GlucData[GlucData['SIt'] > 0]
    GlucData = GlucData[GlucData['SIt+1'] > 0]   
    GlucData = GlucData.reset_index()
    return GlucData

if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    glucData = load_data()
    measured = glucData.loc[:,['SIt', 'SIt+1']].values
    measured = measured[0:1,:]
    sigma = np.load('Sigma_JL.npy')   
    res = 750
    
    grid_pts = [np.linspace(10**-8.5, 10**-1.8, res), np.linspace(10**-8.5, 10**-1.8, res)]
    X_pts, Y_pts = centre_pts(grid_pts, measured)
    
    density_func_raw = Array('d', res**2)
    density_func = np.frombuffer(density_func_raw.get_obj()).reshape((res, res))
    density_func.fill(0)
    start = time()
    with Pool(processes=8, initializer=init_worker, initargs=(Lock(), density_func_raw,  grid_pts, res)) as pool:
        pool.starmap(bivar_norm, [(measured[i], sigma[i,:], X_pts[i], Y_pts[i]) for i in range(len(measured))])
    print(time() - start)
    np.save('C:\\WinPython-64bit-3.5.4.1Qt5\\Glucose\\18-06 2D Kernel (JL)\\PDF_2D_JL', density_func)
    print(Trap2D(density_func))
    
    plt.figure()
    plt.contour(grid_pts[0], grid_pts[1], density_func, 100)