# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 15:48:43 2018

@author: Shaun
"""

import numpy as np
import scipy.integrate

def Interpolate(idx, tgt):
    if linear_prob[idx] < tgt:
        inter = (tgt - linear_prob[idx])/(linear_prob[idx+1]-linear_prob[idx])*grid_pts[idx+2] + (linear_prob[idx+1] - tgt)/(linear_prob[idx+1]-linear_prob[idx])*grid_pts[idx+1]
    else:
        inter = (linear_prob[idx] - tgt)/(linear_prob[idx]-linear_prob[idx-1])*grid_pts[idx] + (tgt - linear_prob[idx-1])/(linear_prob[idx]-linear_prob[idx-1])*grid_pts[idx+1]
    return inter

ab = np.zeros(5)
ib = np.zeros(5)
bb = np.zeros(5)


for j in range(5):
    
    res = 150
    
    grid_pts = np.linspace(-8.5, -1.5, res)
    conf_ints = np.load('PDF_3D_' + str(j+1) + '.npy')
    test_pts = np.load('TEST_3D_' + str(j+1) + '.npy')
    Conf_Int = [0, 0]
    Conf_Int[0] = np.zeros(len(test_pts))
    Conf_Int[1] = np.zeros(len(test_pts))
    in_bounds = np.zeros(len(test_pts))
    ab_bounds = np.zeros(len(test_pts))
    bl_bounds = np.zeros(len(test_pts))
    for i in range(len(test_pts)):
        linear_prob = scipy.integrate.cumtrapz(conf_ints[i,:], grid_pts)/np.trapz(conf_ints[i,:], grid_pts)
        Conf_Int[0][i] = Interpolate(np.argmin(abs(linear_prob - 0.05)), 0.05)
        Conf_Int[1][i] = Interpolate(np.argmin(abs(linear_prob - 0.95)), 0.95)
        if test_pts[i,2] > Conf_Int[0][i] and test_pts[i,2] < Conf_Int[1][i]:
            in_bounds[i] = 1
        elif test_pts[i,2] < Conf_Int[0][i]:
            bl_bounds[i] = 1
        elif test_pts[i,2] > Conf_Int[1][i]:
            ab_bounds[i] = 1

    ab(j) = np.sum(ab_bounds)/len(test_pts)*100
    ib(j) = np.sum(in_bounds)/len(test_pts)*100
    bb(j) = np.sum(bl_bounds)/len(test_pts)*100
     
    #print('Cross Validation for Part ' + str(j+1))
    #print(str(np.sum(in_bounds)/len(test_pts)*100) + '% of points within 95% confidence interval')    
    #print(str(np.sum(ab_bounds)/len(test_pts)*100) + '% of points above 95% confidence interval')    
    #print(str(np.sum(bl_bounds)/len(test_pts)*100) + '% of points below 95% confidence interval')
    
print('Overall Results')
print(str(np.mean(ib)) + '% of points within 95% confidence interval')  
print(str(np.mean(ab)) + '% of points above 95% confidence interval') 
print(str(np.mean(bb)) + '% of points below 95% confidence interval')