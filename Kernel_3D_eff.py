# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 12:06:41 2018

@author: smd118
"""

import pandas as pd
import numpy as np
from time import time
from multiprocessing import Pool, Lock, Array

NUM_THREADS = 8

def init_worker_1(x_pts_, y_pts_, z_pts_, x_out_, y_out_, z_out_, res_, n_pts_):
    global x_pts, y_pts, z_pts, x_out, y_out, z_out, res, n_pts
    x_pts = x_pts_
    y_pts = y_pts_
    z_pts = z_pts_
    x_out = x_out_
    y_out = y_out_
    z_out = z_out_
    res = res_
    n_pts = n_pts_

def init_worker_2(lock_, out_, means_, grid_pts_, trf_, res_, lims_, x_pts_, y_pts_, z_pts_):
    global lock, out, means, grid_pts, trf, res, lims, x_pts, y_pts, z_pts
    lock = lock_
    out = out_
    means = means_
    grid_pts = grid_pts_
    trf = trf_
    res = res_
    lims = lims_
    x_pts = x_pts_
    y_pts = y_pts_
    z_pts = z_pts_
    
def transform_grid(grid_pts, means):
    #Bivariate Normal Distribution for Ortho-Normalised Case (Covariance Matrix is Identity Matrix)
    x_out = np.zeros([len(grid_pts[0]), len(grid_pts[1]), len(grid_pts[2])])
    y_out = np.zeros([len(grid_pts[0]), len(grid_pts[1]), len(grid_pts[2])])
    z_out = np.zeros([len(grid_pts[0]), len(grid_pts[1]), len(grid_pts[2])])
    s = time()
    for x in range(len(grid_pts[0])):
        for y in range(len(grid_pts[1])):
            for z in range(len(grid_pts[2])):
                xyz = np.matmul([grid_pts[0][x]-means[0], grid_pts[1][y]-means[1], grid_pts[2][z]-means[2]], trf)
                x_out[y,x,z] = xyz[0]
                y_out[y,x,z] = xyz[1]
                z_out[y,x,z] = xyz[2]
    print(time()-s)
    return x_out, y_out, z_out
    
def centre_pts(measured_pt, idx):
    Y = np.zeros(res)
    Z = np.zeros([res,res])
    X = np.argmin(abs(x_out[0,:,0]-measured_pt[0]))
    Y[:] = np.argmin(abs(y_out[:,:,0]-measured_pt[1]),0)
    Z[:,:] = np.argmin(abs(z_out-measured_pt[2]),2)
    x_pts_array = np.frombuffer(x_pts.get_obj()).reshape((n_pts))
    y_pts_array = np.frombuffer(y_pts.get_obj()).reshape((n_pts, res))
    z_pts_array = np.frombuffer(z_pts.get_obj()).reshape((n_pts, res, res))
    x_pts_array[idx] = X
    y_pts_array[idx,:] = Y
    z_pts_array[idx,:,:] = Z

def trivar_norm(measured_pt, sigma, i):
    density_func = np.zeros([len(grid_pts[0]), len(grid_pts[1]), len(grid_pts[2])])
    xlim = np.ceil(5*res*sigma/lims[0])
    ylim = 5*res*sigma/lims[1]
    zlim = 5*res*sigma/lims[2]
    for x in range(int(np.max([x_pts[i]-xlim,0])),int(np.min([x_pts[i]+xlim,res]))):
        ydist = np.ceil(ylim*np.sqrt(1-((x-x_pts[i])/xlim)**2))
        for y in range(int(np.max([y_pts[i,x]-ydist, 0])),int(np.min([y_pts[i,x]+ydist, res]))):
            zdist = np.ceil(zlim*np.sqrt(1-((y-y_pts[i,x])/ylim)**2-((x-x_pts[i])/xlim)**2))
            if np.isnan(zdist) == 1:
                zdist = 1
            for z in range(int(np.max([z_pts[i,y,x]-zdist, 0])),int(np.min([z_pts[i,y,x]+zdist, res]))):
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
    glucData = glucData.drop(['Unnamed: 0', 'Unnamed: 0.1', 'Operative', 'Patient', 't0', 'GF'], axis = 1)
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
    measured, measured_trf, trf_det, trf = transform(glucData)
    
    sigma = np.load('Sigma_3D.npy')
    
    res = 150
    
    lims = np.zeros(3)
    lims[0] = abs(np.matmul([-1.5, 0.2, -1.5], trf)[0] - np.matmul([-8.5, 0.2, -1.5], trf)[0])
    lims[1] = abs(np.matmul([-1.5, 0.2, -1.5], trf)[1] - np.matmul([-1.5, 1.4, -1.5], trf)[1])
    lims[2] = abs(np.matmul([-1.5, 0.2, -1.5], trf)[2] - np.matmul([-1.5, 0.2, -8.5], trf)[2])
    grid_pts = [np.linspace(-8.5, -1.5, res), np.linspace(0.2, 1.4, res), np.linspace(-8.5, -1.5, res)]
    means = np.mean(measured, 0)
    x_out, y_out, z_out = transform_grid(grid_pts, means)
        
    x_pts_raw = Array('d', len(measured_trf))
    x_pts = np.frombuffer(x_pts_raw.get_obj()).reshape((len(measured_trf)))
    x_pts.fill(0)    
    y_pts_raw = Array('d', len(measured_trf)*res)
    y_pts = np.frombuffer(y_pts_raw.get_obj()).reshape((len(measured_trf), res))
    y_pts.fill(0)       
    z_pts_raw = Array('d', len(measured_trf)*res*res)
    z_pts = np.frombuffer(z_pts_raw.get_obj()).reshape((len(measured_trf), res, res))
    z_pts.fill(0)    
    
    print('Starting Now')
    s = time()
    with Pool(processes=8, initializer=init_worker_1, initargs=(x_pts_raw, y_pts_raw, z_pts_raw, x_out, y_out, z_out, res, len(measured_trf))) as pool:
        pool.starmap(centre_pts, [(measured_trf[i], i) for i in range(len(measured_trf))])
    print(time() - s)
    density_func_raw = Array('d', res**3)
    density_func = np.frombuffer(density_func_raw.get_obj()).reshape((res, res, res))
    density_func.fill(0)
    start = time()
    with Pool(processes=8, initializer=init_worker_2, initargs=(Lock(), density_func_raw, means, grid_pts, trf, res, lims, x_pts, y_pts, z_pts)) as pool:
        pool.starmap(trivar_norm, [(measured_trf[i], sigma[i], i) for i in range(len(measured_trf))])
    print(time() - start)
    density_func = density_func*trf_det
    #np.save('C:\WinPython-64bit-3.5.4.1Qt5\Glucose\W3X', density_func)
    print(Trap3D(density_func))