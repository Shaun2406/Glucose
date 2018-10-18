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

def init_worker(lock_, out_, grid_pts_, res_):
    global lock, out, grid_pts, res
    lock = lock_
    out = out_
    grid_pts = grid_pts_
    res = res_

def xval_gen(glucData, n):
    CrossValid =  [0, 0, 0, 0, 0]
    for i in range(n):
        CrossValid[i] = np.zeros([0, 5])
    q = 0
    for Patient, Data in glucData.groupby('Patient'):
        q = q+1
        CrossValid[q % n] = np.append(CrossValid[q % n], Data.loc[:,['SIt', 'SIt+' + str(Output), 'sigma_x', 'sigma_y', 'kernel_vol']].values, 0 )
    return CrossValid
    
def bivar_norm(measured_pt, sigma, X_pts, Y_pts, k_vol):
    density_func = np.zeros([len(grid_pts[0]), len(grid_pts[1])])
    x_dist = np.ceil(5*sigma[0]*res/0.016)
    for x in range(int(np.max([X_pts-x_dist,0])),int(np.min([X_pts+x_dist,res]))):
        x_comp = 1/(2*np.pi*sigma[0]*sigma[1])*np.exp(-0.5*((grid_pts[0][x]-measured_pt[0])**2/sigma[0]**2))/k_vol
        y_dist = np.ceil(5*sigma[1]*res/0.016*np.sqrt(1-((x-X_pts)/x_dist)**2))
        for y in range(int(np.max([Y_pts-y_dist, 0])),int(np.min([Y_pts+y_dist, res]))):
            density_func[y,x] = x_comp*np.exp(-0.5*((grid_pts[1][y]-measured_pt[1])**2/sigma[1]**2))
    out_array = np.frombuffer(out.get_obj()).reshape((res, res))
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
    
    sigma = np.load('Sigma_JL_' + str(Output) + 'H.npy')
    sigma = pd.DataFrame({'sigma_x': sigma[:,0], 'sigma_y': sigma[:,1]})
    k_vol = np.load('KNV_JL_' + str(Output) + 'H.npy')
    k_vol = pd.DataFrame({'kernel_vol': k_vol})
    glucData = pd.merge(glucData, sigma, left_index = True, right_index = True)
    glucData = pd.merge(glucData, k_vol, left_index = True, right_index = True)
    
    res = 400
    
    grid_pts = [np.linspace(0, 0.016, res), np.linspace(0, 0.016, res)]
    
    
    for i in range(5):
        print(['Starting Part ' + str(i+1) + ' Now'])        
        start = time()
        input_pts = xval_gen(glucData, 5)
        input_pts[i] = np.zeros([0, 5])
        input_pts = np.concatenate(input_pts)
        #input_pts = input_pts[0:10,:]
        X_pts, Y_pts = centre_pts(grid_pts, input_pts[:,0:2])
        
        density_func_raw = Array('d', res**2)
        density_func = np.frombuffer(density_func_raw.get_obj()).reshape((res, res))
        density_func.fill(0)
        
        with Pool(processes=8, initializer=init_worker, initargs=(Lock(), density_func_raw, grid_pts, res)) as pool:
            pool.starmap(bivar_norm, [(input_pts[j,0:2], input_pts[j,2:4], X_pts[j], Y_pts[j], input_pts[j,4]) for j in range(len(input_pts))])

        np.save('PDF_JL_' + str(Output) + 'H_' + str(i+1), density_func)
        
        print(time() - start)    
        print(Trap2D(density_func))

plt.figure()
plt.contour(grid_pts[0], grid_pts[1], np.log10(density_func), 100)
        
plt.figure()
plt.contour(grid_pts[0], grid_pts[1], density_func, 100)
