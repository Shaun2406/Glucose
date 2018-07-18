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

def Interpolate1(idx, tgt):
    if linear_prob[idx] > tgt:
        inter = grid_pts[idx]
    else:
        inter = grid_pts[idx+1]
    return inter

def Interpolate2(idx, tgt):
    if linear_prob[idx] < tgt:
        inter = grid_pts[idx+2]
    else:
        inter = grid_pts[idx+1]
    return inter

ab = np.zeros(5)
ib = np.zeros(5)
bb = np.zeros(5)
ln = np.zeros(5)

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
        Conf_Int[0][i] = Interpolate1(np.argmin(abs(linear_prob - 0.05)), 0.05)
        Conf_Int[1][i] = Interpolate2(np.argmin(abs(linear_prob - 0.95)), 0.95)
        if 10**test_pts[i,2] > 10**Conf_Int[0][i] and 10**test_pts[i,2] < 10**Conf_Int[1][i]:
            in_bounds[i] = 1
        elif 10**test_pts[i,2] < 10**Conf_Int[0][i]:
            bl_bounds[i] = 1
        elif 10**test_pts[i,2] > 10**Conf_Int[1][i]:
            ab_bounds[i] = 1

    ab[j] = np.sum(ab_bounds)
    ib[j] = np.sum(in_bounds)
    bb[j] = np.sum(bl_bounds)
    ln[j] = len(test_pts)
    #print('Cross Validation for Part ' + str(j+1))
    #print(str(np.sum(in_bounds)/len(test_pts)*100) + '% of points within 95% confidence interval')    
    #print(str(np.sum(ab_bounds)/len(test_pts)*100) + '% of points above 95% confidence interval')    
    #print(str(np.sum(bl_bounds)/len(test_pts)*100) + '% of points below 95% confidence interval')
    
print('Overall Results')
print(str("%.2f" % (np.sum(ib)/np.sum(ln)*100)) + '% of points within 95% confidence interval')  
print(str("%.2f" % (np.sum(ab)/np.sum(ln)*100)) + '% of points above 95% confidence interval') 
print(str("%.2f" % (np.sum(bb)/np.sum(ln)*100)) + '% of points below 95% confidence interval')