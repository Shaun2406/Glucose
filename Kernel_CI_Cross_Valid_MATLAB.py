# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 15:08:49 2018

@author: smd118
"""

import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt

method = 'JL' #3D or JL
res = 400

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

def Trap2D(Arr):
    l = len(Arr)-1
    corners = Arr[0,0]+Arr[l,l]+Arr[0,l]+Arr[l,0]
    edges = np.sum(Arr[1:l,0])+np.sum(Arr[1:l,l])+np.sum(Arr[0,1:l])+np.sum(Arr[l,1:l])
    middle = np.sum(Arr[1:l,1:l])
    tot = (middle*4+edges*2+corners)/4*0.016**2/(res-1)**2
    return tot

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

for interval in range(1,4):
    for cross in range(1,6):        
        bin_size = [13433, 12263, 12365, 12249, 11768]
        bin_size[cross-1] = 0
        volume = np.sum(bin_size)
        PDF_3D = np.load('C:\WinPython-64bit-3.5.4.1Qt5\PDF Results\PDF_' + str(method) + '_' + str(interval) + 'H_' + str(cross) + '.npy')
        if method == '3D':
            print('Percentage Error in 3D is ' + str("%.4f" % ((Trap3D(PDF_3D)-volume)/volume*100)) + '%')
            Gt = np.linspace(0.2, 1.4, res)
            grid_pts = np.linspace(-8.5, -1.5, res)
        elif method == 'JL':
            print('Percentage Error in 2D is ' + str("%.4f" % ((Trap2D(PDF_3D)-volume)/volume*100)) + '%')      
            grid_pts = np.linspace(0, 0.016, res)

        Conf_Int = []
        for i in range(5):
            if method == '3D':
                Conf_Int.append(np.zeros([res,res]))
            elif method == 'JL':
                Conf_Int.append(np.zeros([res]))
        for i in range(res):
            for j in range(res):
                if method == '3D':
                    if np.sum(PDF_3D[j,i,:]) != 0:
                        linear_prob = scipy.integrate.cumtrapz(PDF_3D[j,i,:], grid_pts)/np.trapz(PDF_3D[j,i,:], grid_pts)
                        Conf_Int[0][j,i] = Interpolate(np.argmin(abs(linear_prob - 0.05)), 0.05)
                        Conf_Int[1][j,i] = Interpolate(np.argmin(abs(linear_prob - 0.25)), 0.25)
                        Conf_Int[2][j,i] = Interpolate(np.argmin(abs(linear_prob - 0.50)), 0.50)
                        Conf_Int[3][j,i] = Interpolate(np.argmin(abs(linear_prob - 0.75)), 0.75)
                        Conf_Int[4][j,i] = Interpolate(np.argmin(abs(linear_prob - 0.95)), 0.95)
                elif method == 'JL':
                    if np.sum(PDF_3D[:,i]) != 0:
                        linear_prob = scipy.integrate.cumtrapz(PDF_3D[:,i], grid_pts)/np.trapz(PDF_3D[:,i], grid_pts)                  
                        Conf_Int[0][i] = Interpolate(np.argmin(abs(linear_prob - 0.05)), 0.05)
                        Conf_Int[1][i] = Interpolate(np.argmin(abs(linear_prob - 0.25)), 0.25)
                        Conf_Int[2][i] = Interpolate(np.argmin(abs(linear_prob - 0.50)), 0.50)
                        Conf_Int[3][i] = Interpolate(np.argmin(abs(linear_prob - 0.75)), 0.75)
                        Conf_Int[4][i] = Interpolate(np.argmin(abs(linear_prob - 0.95)), 0.95)
        Conf_Width = 10**Conf_Int[4] - 10**Conf_Int[0]
        
        np.savetxt('C:\\WinPython-64bit-3.5.4.1Qt5\\MATLAB Tables\\Conf_Table_' + str(method) + '_' + str(interval) + 'H_' + str(cross) + '_05', Conf_Int[0], delimiter=",")
        np.savetxt('C:\\WinPython-64bit-3.5.4.1Qt5\\MATLAB Tables\\Conf_Table_' + str(method) + '_' + str(interval) + 'H_' + str(cross) + '_25', Conf_Int[1], delimiter=",")
        np.savetxt('C:\\WinPython-64bit-3.5.4.1Qt5\\MATLAB Tables\\Conf_Table_' + str(method) + '_' + str(interval) + 'H_' + str(cross) + '_50', Conf_Int[2], delimiter=",")
        np.savetxt('C:\\WinPython-64bit-3.5.4.1Qt5\\MATLAB Tables\\Conf_Table_' + str(method) + '_' + str(interval) + 'H_' + str(cross) + '_75', Conf_Int[3], delimiter=",")
        np.savetxt('C:\\WinPython-64bit-3.5.4.1Qt5\\MATLAB Tables\\Conf_Table_' + str(method) + '_' + str(interval) + 'H_' + str(cross) + '_95', Conf_Int[4], delimiter=",")

if method == '3D':
    fig = plt.figure()
    SIt2D, Gt2D = np.meshgrid(grid_pts, Gt)
    plt.contour(SIt2D, Gt2D, np.log(Conf_Width), 100)
    plt.xlabel('Sensitivity, SI(t)')
    plt.ylabel('Glucose, G(t)')
if method == 'JL':
    fig = plt.figure()
    plt.plot(grid_pts, np.log(Conf_Width))
    plt.xlabel('Sensitivity, SI(t)')
    plt.ylabel('Interval Width, SI(t+1)')