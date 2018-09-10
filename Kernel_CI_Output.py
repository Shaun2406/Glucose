# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 12:06:41 2018

@author: smd118
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.integrate
import scipy.interpolate
plt.close("all")

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

res = 400

Output = 3

PDF_3D = np.load('PDF_3D_' + str(Output) + 'H.npy')

print('Percentage Error in 3D is ' + str("%.4f" % ((Trap3D(PDF_3D)-62078)/62078*100)) + '%')

#Dimensions of variables
Gt = np.linspace(0.2, 1.4, res)
grid_pts = np.linspace(-8.5, -1.5, res)
Conf_Int = []
for i in range(2):
    Conf_Int.append(np.zeros([res,res]))
for i in range(res):
    for j in range(res):
        if np.sum(PDF_3D[j,i,:]) != 0:
            linear_prob = scipy.integrate.cumtrapz(PDF_3D[j,i,:], grid_pts)/np.trapz(PDF_3D[j,i,:], grid_pts)
            Conf_Int[0][j,i] = Interpolate(np.argmin(abs(linear_prob - 0.05)), 0.05)
            Conf_Int[1][j,i] = Interpolate(np.argmin(abs(linear_prob - 0.95)), 0.95)
Conf_Width = 10**Conf_Int[1] - 10**Conf_Int[0]
Conf_Table = np.zeros([0,4])
for i in range(len(Gt)):
    Conf_Table = np.append(Conf_Table, np.transpose(np.vstack([Gt[i]*np.ones([len(Gt)]), grid_pts, Conf_Int[0][i,:], Conf_Int[1][i,:]])), 0)

np.savetxt('Conf_Table_3D_' + str(Output) + 'H', Conf_Table, delimiter=",")

fig = plt.figure()
SIt2D, Gt2D = np.meshgrid(grid_pts, Gt)
plt.contour(SIt2D, Gt2D, np.log(Conf_Width), 100)
plt.xlabel('Sensitivity, SI(t)')
plt.ylabel('Glucose, G(t)')