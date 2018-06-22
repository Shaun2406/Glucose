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
                density_func[y,x,z] = 1/((2*np.pi)**(3/2)*sigma**3)*np.exp(-1/2*((xyz[0]-measured_pt[0])**2/sigma**2+(xyz[1]-measured_pt[1])**2/sigma**2+(xyz[2]-measured_pt[2])**2/sigma**2))
    out_array = np.frombuffer(out.get_obj()).reshape((res, res, res))
    lock.acquire()
    try:
        out_array += density_func
    finally:
        lock.release()

def load_data():
    #LOADING DATA
    glucData = pd.read_csv('GlucDataOverall.csv')
    glucData = glucData.drop(['Unnamed: 0', 'Unnamed: 0.1', 'Operative', 'Patient', 't0', 'GF'], axis = 1)
    glucData['Gender'] = glucData['Gender'] == 'female'
    glucData['Gender'] = glucData['Gender'].astype(int)
    
    #LOGGING RELEVANT DATA
    glucData['Gt'] = np.log10(glucData['Gt'])
    glucData = glucData[np.isnan(glucData['Gt']) == 0]
    
    glucData['SIt+1'] = np.log10(glucData['SIt+1'])
    glucData = glucData[np.isnan(glucData['SIt+1']) == 0]
    
    glucData['SIt'] = np.log10(glucData['SIt'])
    glucData = glucData[np.isnan(glucData['SIt']) == 0]
    
    glucData = glucData.reset_index()
    return glucData

def transform(glucData):
    '''Create an Ortho-Normalised Matrix Xdec - 2D, for w(x)'''
    Xin = glucData.loc[:,['SIt', 'Gt', 'SIt+1']].values
    Cin = np.cov(np.transpose(Xin))
    Rin = np.linalg.cholesky(Cin)
    Ain = np.linalg.inv(np.transpose(Rin))
    detAin = np.linalg.det(Ain)
    Xin0 = Xin - np.mean(Xin, 0)
    Xindec = np.matmul(Xin0, Ain)
    
    return (Xin, Xindec, detAin, Ain)

if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    glucData = load_data()
    measured, measured_trf, trf_det, trf = transform(glucData)
    
    sigma = pd.read_csv('KernelSigma.csv')
    sigma = sigma.drop(['Unnamed: 0'], axis = 1)
    sigma = np.array(sigma)
    
    res = 10
    
    grid_pts = [np.linspace(np.min(measured[:,0]), np.max(measured[:,0]), res),
                np.linspace(np.min(measured[:,1]), np.max(measured[:,1]), res),
                np.linspace(np.min(measured[:,2]), np.max(measured[:,2]), res)]
    means = np.mean(measured, 0)

    density_func_raw = Array('d', res**3)
    density_func = np.frombuffer(density_func_raw.get_obj()).reshape((res, res, res))
    density_func.fill(0)
    start = time()
    with Pool(processes=8, initializer=init_worker, initargs=(Lock(), density_func_raw, means, grid_pts, trf, res)) as pool:
        pool.starmap(trivar_norm, [(measured_trf[i], sigma[i][0]) for i in range(len(measured_trf))])
    print(time() - start)
    density_func = density_func*trf_det
    print(np.sum(density_func)/res**3*(np.max(measured[:,0])-np.min(measured[:,0]))*(np.max(measured[:,1])-np.min(measured[:,1]))*(np.max(measured[:,2])-np.min(measured[:,2])))
    np.save('C:\WinPython-64bit-3.5.4.1Qt5\Glucose\W3X', density_func)
