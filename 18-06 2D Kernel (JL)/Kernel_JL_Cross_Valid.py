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

Output = 3

def init_worker(out_, means_, grid_pts_, res_, input_pts_, test_length_):
    global out, means, grid_pts, res, input_pts, test_length
    out = out_
    means = means_
    grid_pts = grid_pts_
    res = res_
    input_pts = input_pts_
    test_length = test_length_
    
def xval_gen(glucData, n):
    CrossValid =  [0, 0, 0, 0, 0]
    for i in range(n):
        CrossValid[i] = np.zeros([0, 5])
    q = 0
    for Patient, Data in glucData.groupby('Patient'):
        q = q+1
        CrossValid[q % n] = np.append(CrossValid[q % n], Data.loc[:,['SIt', 'SIt+' + str(Output), 'sigma_x', 'sigma_y', 'kernel_vol']].values, 0 )
    return CrossValid

def Trap2D(Arr):
    l = len(Arr)-1
    corners = Arr[0,0]+Arr[l,l]+Arr[0,l]+Arr[l,0]
    edges = np.sum(Arr[1:l,0])+np.sum(Arr[1:l,l])+np.sum(Arr[0,1:l])+np.sum(Arr[l,1:l])
    middle = np.sum(Arr[1:l,1:l])
    tot = (middle*4+edges*2+corners)/4*0.016**2/(res-1)**2
    return tot

def pdf_sample(measured_pt, idx):
    density_func = np.zeros(res)
    for i in range(len(input_pts)):
        x_dist = abs(measured_pt[0] - input_pts[i,0])
        if x_dist < 5*input_pts[i,2]:
            y_dist = np.ceil(5*input_pts[i,3]*res/(10**-1.7-10**-8.5)*np.sqrt(1-(x_dist/(5*input_pts[i,2]))**2))
            y_pts = np.argmin(abs(grid_pts - input_pts[i,1]))
            x_comp = 1/(2*np.pi*input_pts[i,2]*input_pts[i,3])*np.exp(-0.5*((measured_pt[0]-input_pts[i,0])**2/input_pts[i,2]**2))/input_pts[i,4]
            for j in range(int(np.max([y_pts - y_dist, 0])), int(np.min([y_pts + y_dist, res]))):
                density_func[j] = density_func[j] + x_comp*np.exp(-0.5*((grid_pts[j]-input_pts[i,1])**2/input_pts[i,3]**2))
    out_array = np.frombuffer(out.get_obj()).reshape((test_length, res))
    out_array[idx,:] = density_func
        
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
    
    glucData = glucData.reset_index()
    return glucData

if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    
    glucData = load_data()
    sigma = np.load('Sigma_JL_' + str(Output) + 'H.npy')
    sigma = pd.DataFrame({'sigma_x': sigma[:,0], 'sigma_y': sigma[:,1]})
    k_vol = np.load('KNV_JL_' + str(Output) + 'H.npy')
    k_vol = pd.DataFrame({'kernel_vol': k_vol})
    glucData = pd.merge(glucData, sigma, left_index = True, right_index = True)
    glucData = pd.merge(glucData, k_vol, left_index = True, right_index = True)
    
    res = 500    
    grid_pts = np.linspace(0, 0.016, res)

    for i in range(5):
        input_pts = xval_gen(glucData, 5)
        test_pts = input_pts[i][:,0:2]
        #test_pts = test_pts[0:10]
        input_pts[i] = np.zeros([0, 5])
        input_pts = np.concatenate(input_pts)
        means = np.mean(input_pts[:,0:2], 0)
        density_func_raw = Array('d', res*len(test_pts))
        density_func = np.frombuffer(density_func_raw.get_obj()).reshape((len(test_pts), res))
        density_func.fill(0)
        
        print(['Starting Part ' + str(i+1) + ' Now'])
        start = time()
        
        with Pool(processes=8, initializer=init_worker, initargs=(density_func_raw, means, grid_pts, res, input_pts, len(test_pts))) as pool:
            pool.starmap(pdf_sample, [(test_pts[j,:], j) for j in range(len(test_pts))])
        print(time() - start)
    
        np.save('C:\\WinPython-64bit-3.5.4.1Qt5\\Glucose\\18-06 2D Kernel (JL)\\CV Results\\PDF_JL_'  + str(Output) + 'H_' + str(i+1), density_func)
        np.save('C:\\WinPython-64bit-3.5.4.1Qt5\\Glucose\\18-06 2D Kernel (JL)\\CV Results\\TEST_JL_' + str(Output) + 'H_' + str(i+1), test_pts)