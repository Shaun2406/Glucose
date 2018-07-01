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
    
def var_norm(measured_pt, sigma):
    density_func = np.zeros(len(grid_pts))
    for x in range(len(grid_pts)):
        density_func[x] = 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-0.5*((grid_pts[x]-measured_pt)**2/sigma**2))
    density_func = density_func/Trap1D(density_func, grid_pts)
    out_array = np.frombuffer(out.get_obj()).reshape(res)
    lock.acquire()
    try:
        out_array += density_func
    finally:
        lock.release()

def Trap1D(Arr, xArr):
    tot = 0
    for i in range(len(xArr)-1):
        tot = tot+(Arr[i]+Arr[i+1])/2*(xArr[i+1]-xArr[i])
    return tot

def Trap1DCum(Arr, xArr):
    tot = np.zeros(len(Arr))
    for i in range(len(xArr)-1):
        tot[i+1] = tot[i]+(Arr[i]+Arr[i+1])/2*(xArr[i+1]-xArr[i])
    return tot

def load_data():
    #Loading and transforming data
    GlucData = pd.read_csv('D:\\Glucose\\GlucDataOverall.csv')
    GlucData = GlucData[GlucData['SIt'] > 0]
    GlucData = GlucData[GlucData['SIt+1'] > 0]   
    GlucData = GlucData.reset_index()
    return GlucData

if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    glucData = load_data()
    measured = glucData.loc[:,['SIt']].values
    sigma = np.load('Sigma_JL.npy')   
    res = 500
    
    #grid_pts = np.logspace(-8.5, -1.5, res)
    grid_pts = np.linspace(10**-8.5, 10**-1.5, res)
    
    density_func_raw = Array('d', res)
    density_func = np.frombuffer(density_func_raw.get_obj()).reshape(res)
    density_func.fill(0)
    start = time()
    with Pool(processes=8, initializer=init_worker, initargs=(Lock(), density_func_raw,  grid_pts, res)) as pool:
        pool.starmap(var_norm, [(measured[i,0], sigma[i,0]) for i in range(len(measured))])
    print(time() - start)
    #np.save('D:\\Glucose\\18-06 2D Kernel (JL)\\PDF_1D_JL', density_func)
    print(Trap1D(density_func, grid_pts))
    plt.figure()
    plt.plot(grid_pts, density_func)
    linear_prob = Trap1DCum(density_func, grid_pts)/Trap1D(density_func, grid_pts)
    print(grid_pts[np.argmin(abs(linear_prob - 0.5))])