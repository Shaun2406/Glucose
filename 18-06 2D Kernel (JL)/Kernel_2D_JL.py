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
    xlim_up = np.argmin(abs(measured_pt[0]+5*sigma[0]-grid_pts[0])) - X_pts
    ylim_up = np.argmin(abs(measured_pt[1]+5*sigma[1]-grid_pts[1])) - Y_pts
    xlim_lw = X_pts - np.argmin(abs(measured_pt[0]-5*sigma[0]-grid_pts[0]))
    ylim_lw = Y_pts - np.argmin(abs(measured_pt[1]-5*sigma[1]-grid_pts[1]))   
    for x in range(int(np.max([X_pts-xlim_lw,0])),int(np.min([X_pts+xlim_up,res]))):
        #y_up = 
        #y_dn = 
        for y in range(int(np.max([Y_pts-ylim_lw,0])),int(np.min([Y_pts+ylim_up,res]))):
            density_func[y,x] = 1/(2*np.pi*sigma[0]*sigma[1])*np.exp(-0.5*((grid_pts[0][x]-measured_pt[0])**2/sigma[0]**2+(grid_pts[1][y]-measured_pt[1])**2/sigma[1]**2))
    #density_func = density_func/Trap2D(density_func, grid_pts[0], grid_pts[1])
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

def Trap2D(Arr, xArr, yArr):
    tot = 0
    for i in range(len(xArr)-1):
        for j in range(len(yArr)-1):
            tot = tot+(Arr[j,i]+Arr[j+1,i+1]+Arr[j+1,i]+Arr[j,i+1])/4*(xArr[i+1]-xArr[i])*(yArr[j+1]-yArr[j])
    return tot

def load_data():
    #Loading and transforming data
    GlucData = pd.read_csv('GlucDataOverall.csv')
    GlucData = GlucData[GlucData['SIt'] > 0]
    GlucData = GlucData[GlucData['SIt+1'] > 0]   
    GlucData = GlucData.reset_index()
    return GlucData

if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    glucData = load_data()
    measured = glucData.loc[:,['SIt', 'SIt+1']].values
    sigma = np.load('Sigma_JL.npy')   
    res = 150
    
    grid_pts = [np.logspace(-8.5, -1.5, res), np.logspace(-8.5, -1.5, res)]
    X_pts, Y_pts = centre_pts(grid_pts, measured)
    
    density_func_raw = Array('d', res**2)
    density_func = np.frombuffer(density_func_raw.get_obj()).reshape((res, res))
    density_func.fill(0)
    start = time()
    with Pool(processes=8, initializer=init_worker, initargs=(Lock(), density_func_raw,  grid_pts, res)) as pool:
        pool.starmap(bivar_norm, [(measured[i], sigma[i,:], X_pts[i], Y_pts[i]) for i in range(len(measured))])
    print(time() - start)
    np.save('PDF_2D_JL_Unscaled', density_func)
    print(Trap2D(density_func, grid_pts[0], grid_pts[1]))
    
    plt.figure()
    plt.contour(grid_pts[0], grid_pts[1], np.log(density_func), 100)