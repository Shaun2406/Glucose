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

Output = 3

def init_worker(lock_, out_, out2_, grid_pts_, res_):
    global lock, out, out2, grid_pts, res
    lock = lock_
    out = out_
    out2 = out2_
    grid_pts = grid_pts_
    res = res_
    
def bivar_norm(measured_pt, sigma, X_pts, Y_pts, i):
    density_func = np.zeros([len(grid_pts[0]), len(grid_pts[1])])
    x_dist = np.ceil(5*sigma[0]*res/0.016)
    for x in range(int(np.max([X_pts-x_dist,0])),int(np.min([X_pts+x_dist,res]))):
        x_comp = 1/(2*np.pi*sigma[0]*sigma[1])*np.exp(-0.5*((grid_pts[0][x]-measured_pt[0])**2/sigma[0]**2))
        y_dist = np.ceil(5*sigma[1]*res/0.016*np.sqrt(1-((x-X_pts)/x_dist)**2))
        for y in range(int(np.max([Y_pts-y_dist, 0])),int(np.min([Y_pts+y_dist, res]))):
            density_func[y,x] = x_comp*np.exp(-0.5*((grid_pts[1][y]-measured_pt[1])**2/sigma[1]**2))
    k_vol = Trap2D(density_func)
    density_func = density_func/k_vol
    out_array = np.frombuffer(out.get_obj()).reshape((res, res))
    out2_array = np.frombuffer(out2.get_obj()).reshape(62078)
    out2_array[i] = k_vol
    lock.acquire()
    try:
        out_array += density_func
    finally:
        lock.release()
        
def centre_pts(grid_pts, measured):
    #Bivariate Normal Distribution for Ortho-Normalised Case (Covariance Matrix is Identity Matrix)
    X_pts = np.zeros(len(measured))
    Y_pts = np.zeros(len(measured))
    for i in range(len(measured)):
        X_pts[i] = np.argmin(abs(grid_pts[0]-measured[i,0]))
        Y_pts[i] = np.argmin(abs(grid_pts[1]-measured[i,1]))
    return X_pts, Y_pts

def Trap2D(Arr):
    l = len(Arr)-1
    corners = Arr[0,0]+Arr[l,l]+Arr[0,l]+Arr[l,0]
    edges = np.sum(Arr[1:l,0])+np.sum(Arr[1:l,l])+np.sum(Arr[0,1:l])+np.sum(Arr[l,1:l])
    middle = np.sum(Arr[1:l,1:l])
    tot = (middle*4+edges*2+corners)/4*0.016**2/(res-1)**2
    return tot

def load_data():
    #Loading and transforming data
    GlucData = pd.read_csv('GlucData3H_Overall.csv')
    GlucData = GlucData[GlucData['SIt'] > 0]
    GlucData = GlucData[GlucData['SIt+1'] > 0]
    GlucData = GlucData[GlucData['SIt+2'] > 0]
    GlucData = GlucData[GlucData['SIt+3'] > 0]
    GlucData = GlucData.reset_index()
    return GlucData

if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    glucData = load_data()
    measured = glucData.loc[:,['SIt', 'SIt+' + str(Output)]].values
    sigma = np.load('Sigma_JL_' + str(Output) + 'H.npy')   
    res = 500
    
    grid_pts = [np.linspace(0, 0.016, res), np.linspace(0, 0.016, res)]
    X_pts, Y_pts = centre_pts(grid_pts, measured)
    
    density_func_raw = Array('d', res**2)
    density_func = np.frombuffer(density_func_raw.get_obj()).reshape((res, res))
    density_func.fill(0)
    
    kernel_vol_raw = Array('d', len(measured))
    kernel_vol = np.frombuffer(kernel_vol_raw.get_obj()).reshape(len(measured))
    kernel_vol.fill(0)
    
    start = time()
    with Pool(processes=8, initializer=init_worker, initargs=(Lock(), density_func_raw, kernel_vol_raw, grid_pts, res)) as pool:
        pool.starmap(bivar_norm, [(measured[i], sigma[i,:], X_pts[i], Y_pts[i], i) for i in range(len(measured))])
    print(time() - start)
    np.save('PDF_JL_' + str(Output) + 'H', density_func)
    np.save('KNV_JL_' + str(Output) + 'H', kernel_vol)
    print(Trap2D(density_func))
    
    plt.figure()
    plt.contour(grid_pts[0], grid_pts[1], np.log10(density_func), 100)
    
    plt.figure()
    plt.contour(grid_pts[0], grid_pts[1], density_func, 100)