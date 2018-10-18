# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 12:06:41 2018

@author: smd118

3D Method Broken Into Five Sets, Outputs PDF Field
"""

import pandas as pd
import numpy as np
from time import time
from multiprocessing import Pool, Lock, Array

NUM_THREADS = 8

Output = 3

def init_worker(lock_, out_, means_, grid_pts_, trf_, res_, lims_, x_out_, y_out_, z_out_):
    global lock, out, means, grid_pts, trf, res, lims, x_out, y_out, z_out
    lock = lock_
    out = out_
    means = means_
    grid_pts = grid_pts_
    trf = trf_
    res = res_
    lims = lims_
    x_out = x_out_
    y_out = y_out_
    z_out = z_out_

def xval_gen(glucData, n):
    CrossValid =  [0, 0, 0, 0, 0]
    for i in range(n):
        CrossValid[i] = np.zeros([0, 4])
    q = 0
    for Patient, Data in glucData.groupby('Patient'):
        q = q+1
        CrossValid[q % n] = np.append(CrossValid[q % n], Data.loc[:,['SIt', 'Gt', 'SIt+' + str(Output), 'Sigma']].values, 0 )
    return CrossValid
    
def transform_grid(grid_pts, means):
    #Bivariate Normal Distribution for Ortho-Normalised Case (Covariance Matrix is Identity Matrix)
    x_out = np.zeros([len(grid_pts[1])])
    y_out = np.zeros([len(grid_pts[0]), len(grid_pts[1])])
    z_out = np.zeros([len(grid_pts[0]), len(grid_pts[1]), len(grid_pts[2])])
    s = time()
    for x in range(len(grid_pts[0])):
        for y in range(len(grid_pts[1])):
            for z in range(len(grid_pts[2])):
                xyz = np.matmul([grid_pts[0][x]-means[0], grid_pts[1][y]-means[1], grid_pts[2][z]-means[2]], trf)
                if y == 0 and z == 0:
                    x_out[x] = xyz[0]
                if z == 0:
                    y_out[y,x] = xyz[1]
                z_out[y,x,z] = xyz[2]
    print(time()-s)
    return x_out, y_out, z_out
    
def trivar_norm(measured_pt, sigma):
    density_func = np.zeros([res, res, res])
    y_pts = np.zeros([res])
    z_pts = np.zeros([res,res])
    x_pts = np.argmin(abs(x_out[:]-measured_pt[0]))
    y_pts[:] = np.argmin(abs(y_out[:,:]-measured_pt[1]),0)
    z_pts[:,:] = np.argmin(abs(z_out-measured_pt[2]),2)
    xlim = np.ceil(5*res*sigma/lims[0])
    ylim = 5*res*sigma/lims[1]
    zlim = 5*res*sigma/lims[2]
    
    for x in range(int(np.max([x_pts-xlim,0])),int(np.min([x_pts+xlim,res]))):
        ydist = np.ceil(ylim*np.sqrt(1-((x-x_pts)/xlim)**2))
        x_in = np.matmul([grid_pts[0][x]-means[0], 0, 0], trf)
        x_comp = 1/((2*np.pi)**1.5*sigma**3)*np.exp(-0.5*((x_in[0]-measured_pt[0])/sigma)**2)
        for y in range(int(np.max([y_pts[x]-ydist, 0])),int(np.min([y_pts[x]+ydist, res]))):
            zdist = np.ceil(zlim*np.sqrt(1-((y-y_pts[x])/ylim)**2-((x-x_pts)/xlim)**2))
            y_in = np.matmul([grid_pts[0][x]-means[0], grid_pts[1][y]-means[1], 0], trf)
            y_comp = np.exp(-0.5*((y_in[1]-measured_pt[1])/sigma)**2)           
            if np.isnan(zdist) == 1:
                zdist = 1
            for z in range(int(np.max([z_pts[y,x]-zdist, 0])),int(np.min([z_pts[y,x]+zdist, res]))):
                z_in = np.matmul([grid_pts[0][x]-means[0], grid_pts[1][y]-means[1], grid_pts[2][z]-means[2]], trf)
                density_func[y,x,z] = x_comp*y_comp*np.exp(-0.5*((z_in[2]-measured_pt[2])/sigma)**2)
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
    glucData = pd.read_csv('GlucData3H_Overall.csv')
    glucData = glucData.drop(['Unnamed: 0', 'Operative', 't0', 'GF'], axis = 1)
    glucData['Gender'] = glucData['Gender'] == 'female'
    glucData['Gender'] = glucData['Gender'].astype(int)
    
    #LOGGING RELEVANT DATA
    glucData = glucData[glucData['SIt'] > 0]
    glucData = glucData[glucData['SIt+1'] > 0]
    glucData = glucData[glucData['SIt+2'] > 0]
    glucData = glucData[glucData['SIt+3'] > 0]
    
    glucData['Gt'] = np.log10(glucData['Gt'])
    glucData['SIt'] = np.log10(glucData['SIt'])
    glucData['SIt+' + str(Output)] = np.log10(glucData['SIt+' + str(Output)])
    
    glucData = glucData.reset_index()
    return glucData

def transform(X):
    '''Create an Ortho-Normalised Matrix Xdec - 2D, for w(x)'''
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
    sigma = np.load('Sigma_3D_' + str(Output) + 'H.npy')
    sigma = pd.DataFrame({'Sigma': sigma})
    glucData = pd.merge(glucData, sigma, left_index = True, right_index = True)
       
    res = 400
    grid_pts = [np.linspace(-8.5, -1.5, res), np.linspace(0.2, 1.4, res), np.linspace(-8.5, -1.5, res)] 
    
    for i in range(5):
        print(['Starting Part ' + str(i+1) + ' Now'])
        
        start = time()
        
        input_pts = xval_gen(glucData, 5)
        input_pts[i] = np.zeros([0, 4])
        input_pts = np.concatenate(input_pts)
        
        measured, measured_trf, trf_det, trf = transform(input_pts[:,0:3])
        
        lims = np.zeros(3)
        lims[0] = abs(np.matmul([-1.5, 0.2, -1.5], trf)[0] - np.matmul([-8.5, 0.2, -1.5], trf)[0])
        lims[1] = abs(np.matmul([-1.5, 0.2, -1.5], trf)[1] - np.matmul([-1.5, 1.4, -1.5], trf)[1])
        lims[2] = abs(np.matmul([-1.5, 0.2, -1.5], trf)[2] - np.matmul([-1.5, 0.2, -8.5], trf)[2])

        means = np.mean(measured, 0)
        x_out, y_out, z_out = transform_grid(grid_pts, means)
        
        density_func_raw = Array('d', res**3)
        density_func = np.frombuffer(density_func_raw.get_obj()).reshape((res, res, res))
        density_func.fill(0)
        
        with Pool(processes=8, initializer=init_worker, initargs=(Lock(), density_func_raw, means, grid_pts, trf, res, lims, x_out, y_out, z_out)) as pool:
            pool.starmap(trivar_norm, [(measured_trf[i], input_pts[i,3]) for i in range(len(measured_trf))])
        
        density_func = density_func*trf_det
        
        print(time() - start)
        
        np.save('..//PDF_3D_'  + str(Output) + 'H_' + str(i+1), density_func)     