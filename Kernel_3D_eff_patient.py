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

def transform_pts(grid_pts, means):
    x_trf = np.zeros([len(grid_pts[0]), len(grid_pts[1]), len(grid_pts[2])])
    y_trf = np.zeros([len(grid_pts[0]), len(grid_pts[1]), len(grid_pts[2])])
    z_trf = np.zeros([len(grid_pts[0]), len(grid_pts[1]), len(grid_pts[2])])
    for x in range(len(grid_pts[0])):
        for y in range(len(grid_pts[1])):
            for z in range(len(grid_pts[2])):
                xyz = np.matmul([grid_pts[0][x]-means[0], grid_pts[1][y]-means[1], grid_pts[2][z]-means[2]], trf)
                x_trf[y,x,z] = xyz[0]
                y_trf[y,x,z] = xyz[1]
                z_trf[y,x,z] = xyz[2]
    return x_trf, y_trf, z_trf

def centre_pts(x_trf, y_trf, z_trf, measured_trf, means):
    X_pts = np.zeros(len(measured_trf))
    Y_pts = np.zeros([len(measured_trf), len(grid_pts[1])])
    Z_pts = np.zeros([len(measured_trf), len(grid_pts[1]), len(grid_pts[2])])
    for i in range(len(measured_trf)):
        X_pts[i] = np.argmin(abs(x_trf[0,:,0]-measured_trf[i,0]))
        Y_pts[i,:] = np.argmin(abs(y_trf[:,:,0]-measured_trf[i,1]),0)
        Z_pts[i,:,:] = np.argmin(abs(z_trf-measured_trf[i,2]),2)
    return X_pts, Y_pts, Z_pts 

def trivar_norm(measured_pt, sigma, X_pts, Y_pts, Z_pts):
    density_func = np.zeros([len(grid_pts[0]), len(grid_pts[1]), len(grid_pts[2])])
    for x in range(int(np.max([X_pts-18,0])),int(np.min([X_pts+18,150]))):
        for y in range(int(np.max([Y_pts[x]-18,0])),int(np.min([Y_pts[x]+18,150]))):
            for z in range(int(np.max([Z_pts[y,x]-18,0])),int(np.min([Z_pts[y,x]+18,150]))):
                xyz = np.matmul([grid_pts[0][x]-means[0], grid_pts[1][y]-means[1], grid_pts[2][z]-means[2]], trf)
                density_func[y,x,z] = 1/((2*np.pi)**1.5*sigma**3)*np.exp(-0.5*((xyz[0]-measured_pt[0])**2+(xyz[1]-measured_pt[1])**2+(xyz[2]-measured_pt[2])**2)/sigma**2)
    out_array = np.frombuffer(out.get_obj()).reshape((res, res, res))
    lock.acquire()
    try:
        out_array += density_func
    finally:
        lock.release()
        
def Trap3D(Arr):
    l = len(Arr)-1
    corners = Arr[0,0,0]+Arr[0,0,l]+Arr[0,l,0]+Arr[l,0,0]+Arr[0,l,l]+Arr[l,0,l]+Arr[l,l,0]+Arr[l,l,l]
    edges1 = np.sum(Arr[1:l,0,0])+np.sum(Arr[1:l,0,l])+np.sum(Arr[1:l,l,0])+np.sum(Arr[1:l,l,l])
    edges2 = np.sum(Arr[0,1:l,0])+np.sum(Arr[0,1:l,l])+np.sum(Arr[l,1:l,0])+np.sum(Arr[l,1:l,l])
    edges3 = np.sum(Arr[0,0,1:l])+np.sum(Arr[0,l,1:l])+np.sum(Arr[l,0,1:l])+np.sum(Arr[l,l,1:l])
    edges = edges1+edges2+edges3
    faces = np.sum(Arr[0,1:l,1:l])+np.sum(Arr[1:l,0,1:l])+np.sum(Arr[1:l,1:l,0])+np.sum(Arr[l,1:l,1:l])+np.sum(Arr[1:l,l,1:l])+np.sum(Arr[1:l,1:l,l])
    middle = np.sum(Arr[1:l,1:l,1:l])
    tot = (middle*8+faces*4+edges*2+corners)/8*58.8/(res-1)**3
    return tot

def load_data():
    #LOADING DATA
    glucData = pd.read_csv('GlucDataOverall.csv')
    glucData = glucData.drop(['Unnamed: 0', 'Unnamed: 0.1', 'Operative', 't0', 'GF'], axis = 1)
    glucData['Gender'] = glucData['Gender'] == 'female'
    glucData['Gender'] = glucData['Gender'].astype(int)
    
    #LOGGING RELEVANT DATA
    glucData = glucData[glucData['SIt'] > 0]
    glucData = glucData[glucData['SIt+1'] > 0]
    
    glucData['Gt'] = np.log10(glucData['Gt'])
    glucData['SIt+1'] = np.log10(glucData['SIt+1'])
    glucData['SIt'] = np.log10(glucData['SIt'])
    
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

if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    glucData = load_data()
    measured_all, _, trf_det, trf = transform(glucData)
    
    sigma = np.load('Sigma.npy')
    
    res = 150
    
    grid_pts = [np.linspace(-8.5, -1.5, res), np.linspace(0.2, 1.4, res), np.linspace(-8.5, -1.5, res)]
    means = np.mean(measured_all, 0)
    x_trf, y_trf, z_trf = transform_pts(grid_pts, means)

    start_all = time()
    for Patient, Data in glucData.groupby('Patient'):
        measured = Data.loc[:,['SIt', 'Gt', 'SIt+1']].values
        measured_trf = np.matmul(measured-means, trf)
        X_pts, Y_pts, Z_pts = centre_pts(x_trf, y_trf, z_trf, measured_trf, means)
        
        density_func_raw = Array('d', res**3)
        density_func = np.frombuffer(density_func_raw.get_obj()).reshape((res, res, res))
        density_func.fill(0)
        start = time()
        
        with Pool(processes=8, initializer=init_worker, initargs=(Lock(), density_func_raw, means, grid_pts, trf, res)) as pool:
            pool.starmap(trivar_norm, [(measured_trf[i], sigma[i], X_pts[i], Y_pts[i,:], Z_pts[i,:,:]) for i in range(len(measured_trf))])
        print(time() - start)
        density_func = density_func*trf_det
        np.save('C:\\WinPython-64bit-3.5.4.1Qt5\\Glucose\\Patients\\' + str(Patient), density_func)
        density_func = []
    print('Completed All Patients')
    print(time() - start_all)