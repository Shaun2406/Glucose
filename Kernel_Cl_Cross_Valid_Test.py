# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 15:48:43 2018

@author: Shaun
"""

import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
import pandas as pd
plt.close("all")


Output = 2

Prob = 0.9

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

def Interpolate_Grid(idx, tgt):
    if linear_prob[idx] < tgt:
        if linear_prob[idx+1] != linear_prob[idx]:
            f = (tgt - linear_prob[idx])/(linear_prob[idx+1]-linear_prob[idx])
            inter = f*(idx+2) + (1-f)*(idx+1)  
        else:
            f = (tgt - linear_prob[idx-1])/(linear_prob[idx]-linear_prob[idx-1])
            inter = f*(idx+1) + (1-f)*idx
    else:
        if linear_prob[idx] != linear_prob[idx-1]:                                                         
            f = (linear_prob[idx] - tgt)/(linear_prob[idx]-linear_prob[idx-1])  
            inter = f*idx + (1-f)*(idx+1)                   
        else:                                          
            f = (linear_prob[idx+1] - tgt)/(linear_prob[idx+1]-linear_prob[idx])  
            inter = f*(idx+1) + (1-f)*(idx+2)
    return inter

def Locate_Target(idx, tgt):
    if grid_pts[idx] < tgt:
        f = (tgt - grid_pts[idx])/(grid_pts[idx+1]-grid_pts[idx])
        inter = f*(idx+1) + (1-f)*idx
    else:
        f = (grid_pts[idx] - tgt)/(grid_pts[idx]-grid_pts[idx-1])
        inter = f*(idx-1) + (1-f)*idx
    return inter
   
test_length = 62078
plow = (1-Prob)/2
phigh = 1 - plow
ab = np.zeros(5)
ib = np.zeros(5)
bb = np.zeros(5)
ln = np.zeros(5)
q = -1
Cell_Loc = [0, 0]
Cell_Loc[0] = np.zeros(test_length)
Cell_Loc[1] = np.zeros(test_length)
CI_Loc = np.zeros(test_length)
ab_points = np.zeros([0,3])
ib_points = np.zeros([0,3])
bl_points = np.zeros([0,3])
for j in range(5):
    
    res = 300
    
    grid_pts = np.linspace(-8.5, -1.5, res)
    conf_ints = np.load('C:\\WinPython-64bit-3.5.4.1Qt5\\Glucose\\CV Results\\PDF_3D_' + str(Output) + 'H_' + str(j+1) + '.npy')
    test_pts = np.load('C:\\WinPython-64bit-3.5.4.1Qt5\\Glucose\\CV Results\\TEST_3D_' + str(Output) + 'H_' + str(j+1) + '.npy')
    Conf_Int = [0, 0]

    Conf_Int[0] = np.zeros(len(test_pts))
    Conf_Int[1] = np.zeros(len(test_pts))

    in_bounds = np.zeros(len(test_pts))
    ab_bounds = np.zeros(len(test_pts))
    bl_bounds = np.zeros(len(test_pts))
    for i in range(len(test_pts)):
        q = q+1
        linear_prob = scipy.integrate.cumtrapz(conf_ints[i,:], grid_pts)/np.trapz(conf_ints[i,:], grid_pts)
        Conf_Int[0][i] = Interpolate(np.argmin(abs(linear_prob - plow)), plow)
        Conf_Int[1][i] = Interpolate(np.argmin(abs(linear_prob - phigh)), phigh)
        Cell_Loc[0][q] = Interpolate_Grid(np.argmin(abs(linear_prob - plow)), plow)
        Cell_Loc[1][q] = Interpolate_Grid(np.argmin(abs(linear_prob - phigh)), phigh)
        CI_Loc[q] = Locate_Target(np.argmin(abs(grid_pts-test_pts[i,2])), test_pts[i,2])
        if 10**test_pts[i,2] >= 10**Conf_Int[0][i] and 10**test_pts[i,2] <= 10**Conf_Int[1][i]:
            in_bounds[i] = 1
        elif 10**test_pts[i,2] < 10**Conf_Int[0][i]:
            bl_bounds[i] = 1
        elif 10**test_pts[i,2] > 10**Conf_Int[1][i]:
            ab_bounds[i] = 1

    ab[j] = np.sum(ab_bounds)
    ib[j] = np.sum(in_bounds)
    bb[j] = np.sum(bl_bounds)
    ln[j] = len(test_pts)
    
    ab_points = np.append(ab_points, test_pts[ab_bounds == 1], 0)
    ib_points = np.append(ib_points, test_pts[in_bounds == 1], 0)
    bl_points = np.append(bl_points, test_pts[bl_bounds == 1], 0)
    
print('Overall Results')
print(str("%.2f" % (np.sum(ib)/np.sum(ln)*100)) + '% of points within ' + str(100*Prob) + '% confidence interval')  
print(str("%.2f" % (np.sum(ab)/np.sum(ln)*100)) + '% of points above ' + str(100*Prob) + '% confidence interval') 
print(str("%.2f" % (np.sum(bb)/np.sum(ln)*100)) + '% of points below ' + str(100*Prob) + '% confidence interval')

#Above = pd.DataFrame({'SIt': ab_points[:,0], 'Gt': ab_points[:,1], 'SIt1': ab_points[:,2]})
#Below = pd.DataFrame({'SIt': bl_points[:,0], 'Gt': bl_points[:,1], 'SIt1': bl_points[:,2]})
#Within = pd.DataFrame({'SIt': ib_points[:,0], 'Gt': ib_points[:,1], 'SIt1': ib_points[:,2]})
#pd.DataFrame.hist(Above, 'SIt', bins = 50)
#pd.DataFrame.hist(Below, 'SIt', bins = 50)
#pd.DataFrame.hist(Within, 'SIt', bins = 50)

#np.savetxt("3D_Above", ab_points, delimiter=",")
#np.savetxt("3D_Below", bl_points, delimiter=",")
#np.savetxt("3D_Within", ib_points, delimiter=",")

'''q1 = 0
q2 = 0
above_dist = np.zeros(int(np.sum(ab)))
below_dist = np.zeros(int(np.sum(bb)))
for i in range(test_length):
    if CI_Loc[i] < Cell_Loc[0][i]:
        below_dist[q1] = Cell_Loc[0][i] - CI_Loc[i]
        q1 = q1+1
    elif CI_Loc[i] > Cell_Loc[1][i]:
        above_dist[q2] = CI_Loc[i] - Cell_Loc[1][i]
        q2 = q2+1
        
plt.figure()
plt.plot(np.sort(below_dist))
plt.figure()
plt.plot(np.sort(above_dist))'''