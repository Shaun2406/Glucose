# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 12:06:41 2018

Confidence interval and expected values for JL kernel method
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import scipy.interpolate
plt.close("all")

def Trap2D(Arr):
    l = len(Arr)-1
    corners = Arr[0,0]+Arr[l,l]+Arr[0,l]+Arr[l,0]
    edges = np.sum(Arr[1:l,0])+np.sum(Arr[1:l,l])+np.sum(Arr[0,1:l])+np.sum(Arr[l,1:l])
    middle = np.sum(Arr[1:l,1:l])
    tot = (middle*4+edges*2+corners)/4*0.016**2/(res-1)**2
    return tot

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

res = 500

#Loads pre-summed probability fields
PDF_2D = np.load('PDF_2D_JL.npy')
PDF_1D = np.load('PDF_1D_JL.npy')

#Dimensions of variables
grid_pts = np.linspace(0, 0.016, res)
print('Percentage Error in 2D is ' + str((Trap2D(PDF_2D)-62589)/62589*100) + '%')

Conf_Int = [0, 0]
Conf_Int[0] = np.zeros([res])
Conf_Int[1] = np.zeros([res])
SIt_Med = np.zeros([res])
for i in range(res):
        if np.sum(PDF_2D[:,i]) != 0:
            linear_prob = scipy.integrate.cumtrapz(PDF_2D[:,i], grid_pts)/np.trapz(PDF_2D[:,i], grid_pts)
            Conf_Int[0][i] = Interpolate(np.argmin(abs(linear_prob - 0.05)), 0.05)
            Conf_Int[1][i] = Interpolate(np.argmin(abs(linear_prob - 0.95)), 0.95)
            SIt_Med[i] = Interpolate(np.argmin(abs(linear_prob - 0.5)), 0.5)
Conf_Width = Conf_Int[1] - Conf_Int[0]
Conf_Exp_Val = np.trapz(Conf_Width*PDF_1D/62589, grid_pts)
SIt1_Exp_Val = np.trapz(SIt_Med*PDF_1D/62589, grid_pts)
Conf_Up_Val = np.trapz(Conf_Int[1]*PDF_1D/62589, grid_pts)
Conf_Low_Val = np.trapz(Conf_Int[0]*PDF_1D/62589, grid_pts)
print('Expected Confidence Interval width is ' + str(Conf_Exp_Val))
print('Expected Value of SI(t+1) is ' + str(SIt1_Exp_Val) + ' with lower bound ' + str(Conf_Low_Val) + ' and upper bound ' + str(Conf_Up_Val))

np.save('JL_Conf_Width', Conf_Width)
np.save('JL_Conf_Int', Conf_Int)

Conf_JL = [0,0]
res = 300
Gt = np.linspace(0.2, 1.4, res)
grid_pts = np.linspace(-8.5, -1.5, res)

Conf_Width_JL = np.transpose(Conf_Int[0])*np.ones([500,500])
JL_Conf = scipy.interpolate.interp2d(np.linspace(0, 0.016, 500), np.linspace(10**0.2, 10**1.4, 500), Conf_Width_JL)
Conf_JL[0] = JL_Conf.__call__(10**grid_pts, 10**Gt)

Conf_Width_JL = np.transpose(Conf_Int[1])*np.ones([500,500])
JL_Conf = scipy.interpolate.interp2d(np.linspace(0, 0.016, 500), np.linspace(10**0.2, 10**1.4, 500), Conf_Width_JL)
Conf_JL[1] = JL_Conf.__call__(10**grid_pts, 10**Gt)

np.savetxt("Conf_LB_JL", Conf_JL[0], delimiter=",")
np.savetxt("Conf_UB_JL", Conf_JL[1], delimiter=",")