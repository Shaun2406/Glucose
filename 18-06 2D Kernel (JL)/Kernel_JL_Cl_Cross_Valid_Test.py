# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 15:48:43 2018

@author: Shaun
"""

import numpy as np
import scipy.integrate

Output = 3

Prob = 0.5

def Interpolate(idx, tgt):
    if linear_prob[idx] < tgt:
        if linear_prob[idx+1] != linear_prob[idx]:
            f = (tgt - linear_prob[idx])/(linear_prob[idx+1]-linear_prob[idx])
            inter = f*grid_pts[idx+2] + (1-f)*grid_pts[idx+1]   
        else:
            f = (tgt - linear_prob[idx-1])/(linear_prob[idx]-linear_prob[idx-1])
            inter = f*grid_pts[idx+1] + (1-f)*grid_pts[idx]
    else:
        if linear_prob[idx] != linear_prob[idx-1]:                                                         
            f = (linear_prob[idx] - tgt)/(linear_prob[idx]-linear_prob[idx-1])  
            inter = f*grid_pts[idx] + (1-f)*grid_pts[idx+1]                    
        else:                                          
            f = (linear_prob[idx+1] - tgt)/(linear_prob[idx+1]-linear_prob[idx])  
            inter = f*grid_pts[idx+1] + (1-f)*grid_pts[idx+2]
    return inter

ab = np.zeros(5)
ib = np.zeros(5)
bb = np.zeros(5)
ln = np.zeros(5)

ab_points = np.zeros([0,3])
ib_points = np.zeros([0,3])
bl_points = np.zeros([0,3])

plow = (1-Prob)/2
phigh = 1 - plow

for j in range(5):
    
    res = 500
    
    grid_pts = np.linspace(0, 0.016, res)
    conf_ints = np.load('C:\\WinPython-64bit-3.5.4.1Qt5\\Glucose\\18-06 2D Kernel (JL)\\CV Results\\PDF_JL_'  + str(Output) + 'H_' + str(j+1) + '.npy')
    test_pts = np.load('C:\\WinPython-64bit-3.5.4.1Qt5\\Glucose\\18-06 2D Kernel (JL)\\CV Results\\TEST_JL_' + str(Output) + 'H_' + str(j+1) + '.npy')
    Conf_Int = [0, 0]
    Conf_Int[0] = np.zeros(len(test_pts))
    Conf_Int[1] = np.zeros(len(test_pts))
    in_bounds = np.zeros(len(test_pts))
    ab_bounds = np.zeros(len(test_pts))
    bl_bounds = np.zeros(len(test_pts))
    for i in range(len(test_pts)):
        linear_prob = scipy.integrate.cumtrapz(conf_ints[i,:], grid_pts)/np.trapz(conf_ints[i,:], grid_pts)
        Conf_Int[0][i] = Interpolate(np.argmin(abs(linear_prob - plow)), plow)
        Conf_Int[1][i] = Interpolate(np.argmin(abs(linear_prob - phigh)), phigh)
        if test_pts[i,1] > Conf_Int[0][i] and test_pts[i,1] < Conf_Int[1][i]:
            in_bounds[i] = 1
        elif test_pts[i,1] < Conf_Int[0][i]:
            bl_bounds[i] = 1
        elif test_pts[i,1] > Conf_Int[1][i]:
            ab_bounds[i] = 1

    ab[j] = np.sum(ab_bounds)
    ib[j] = np.sum(in_bounds)
    bb[j] = np.sum(bl_bounds)
    ln[j] = len(test_pts)
    
    #ab_points = np.append(ab_points, test_pts[ab_bounds == 1], 0)
    #ib_points = np.append(ib_points, test_pts[in_bounds == 1], 0)
    #bl_points = np.append(bl_points, test_pts[bl_bounds == 1], 0)
    
print('Overall Results')
print(str("%.2f" % (np.sum(ib)/np.sum(ln)*100)) + '% of points within ' + str(100*Prob) + '% confidence interval')  
print(str("%.2f" % (np.sum(ab)/np.sum(ln)*100)) + '% of points above ' + str(100*Prob) + '% confidence interval') 
print(str("%.2f" % (np.sum(bb)/np.sum(ln)*100)) + '% of points below ' + str(100*Prob) + '% confidence interval')

#np.savetxt("2D_Above", ab_points, delimiter=",")
#np.savetxt("2D_Below", bl_points, delimiter=",")
#np.savetxt("2D_Within", ib_points, delimiter=",")