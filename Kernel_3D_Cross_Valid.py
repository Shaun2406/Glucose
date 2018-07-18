# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 12:06:41 2018

@author: smd118
"""

import pandas as pd
import numpy as np
from time import time
from multiprocessing import Pool, Array
import matplotlib.pyplot as plt
plt.close("all")

NUM_THREADS = 8

def init_worker(out_, means_, grid_pts_, trf_, res_, input_pts_, test_length_, zscale_):
    global out, means, grid_pts, trf, res, input_pts, test_length, zscale
    out = out_
    means = means_
    grid_pts = grid_pts_
    trf = trf_
    res = res_
    input_pts = input_pts_
    test_length = test_length_
    zscale = zscale_
    
def xval_gen(glucData, n):
    CrossValid =  [0, 0, 0, 0, 0]
    for i in range(n):
        CrossValid[i] = np.zeros([0, 4])
    q = 0
    for Patient, Data in glucData.groupby('Patient'):
        q = q+1
        CrossValid[q % n] = np.append(CrossValid[q % n], Data.loc[:,['SIt', 'Gt', 'SIt+1', 'Sigma']].values, 0 )
    return CrossValid
        
def pdf_sample(measured_pt, idx):
    sample_pts = np.matmul(np.transpose([np.ones(res)*(measured_pt[0]-means[0]), np.ones(res)*(measured_pt[1]-means[1]), grid_pts-means[2]]), trf)
    measured_pt = np.matmul(measured_pt-means, trf)
    density_func = np.zeros(res)
    for i in range(len(input_pts)):
        xy_dist = np.linalg.norm(np.append(measured_pt[0:2], input_pts[i,2])-input_pts[i, 0:3])
        if xy_dist < 5*input_pts[i,3]:
            z_pts = np.argmin(abs(sample_pts[:,2] - input_pts[i,2]))
            z_dist = zscale*np.sqrt((5*input_pts[i,3])**2-xy_dist**2)
            for j in range(int(np.max([z_pts - z_dist, 0])), int(np.min([z_pts + z_dist, res]))):
                density_func[j] = density_func[j]+1/((2*np.pi)**1.5*input_pts[i,3]**3)*np.exp(-0.5*((sample_pts[j,0]-input_pts[i,0])**2+(sample_pts[j,1]-input_pts[i,1])**2+(sample_pts[j,2]-input_pts[i,2])**2)/input_pts[i,3]**2)
    out_array = np.frombuffer(out.get_obj()).reshape((test_length, res))
    out_array[idx,:] = density_func
        
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
    sigma = np.load('Sigma_3D.npy')
    sigma = pd.DataFrame({'Sigma': sigma})
    glucData = pd.merge(glucData, sigma, left_index = True, right_index = True)
    res = 150    
    grid_pts = np.linspace(-8.5, -1.5, res)

    for i in range(5):
        input_pts = xval_gen(glucData, 5)
        test_pts = input_pts[i][:,0:3]
        input_pts[i] = np.zeros([0, 4])
        input_pts = np.concatenate(input_pts)
        measured, measured_trf, trf_det, trf = transform(input_pts[:,0:3])
        means = np.mean(measured, 0)
        input_pts[:, 0:3] = measured_trf
        zscale = res/abs(np.matmul([-1.5, 0.2, -1.5], trf)[2] - np.matmul([-1.5, 0.2, -8.5], trf)[2])        

        density_func_raw = Array('d', res*len(test_pts))
        density_func = np.frombuffer(density_func_raw.get_obj()).reshape((len(test_pts), res))
        density_func.fill(0)
        
        print(['Starting Part ' + str(i+1) + ' Now'])
        start = time()
        
        with Pool(processes=8, initializer=init_worker, initargs=(density_func_raw, means, grid_pts, trf, res, input_pts, len(test_pts), zscale)) as pool:
            pool.starmap(pdf_sample, [(test_pts[j,:], j) for j in range(len(test_pts))])
        print(time() - start)
        density_func = density_func*trf_det
        np.save('PDF_3D_' + str(i+1), density_func)
        np.save('TEST_3D_' + str(i+1), test_pts)