# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 12:06:41 2018

@author: smd118
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from mpl_toolkits.mplot3d import Axes3D
import os
plt.close("all")

def Trap1D(Arr, xArr):
    tot = 0
    for i in range(len(xArr)-1):
        tot = tot+(Arr[i]+Arr[i+1])/2*(xArr[i+1]-xArr[i])
    return tot

def Trap1DCum(Arr, xArr):
    tot = np.zeros(len(Arr))
    for i in range(len(xArr)-1):
        tot[i+1] = tot[i]+(Arr[i]+Arr[i+1])/2*(xArr[i+1]-xArr[i])
    return tot

def Trap2D(Arr):
    l = len(Arr)-1
    corners = Arr[0,0]+Arr[l,l]+Arr[0,l]+Arr[l,0]
    edges = np.sum(Arr[1:l,0])+np.sum(Arr[1:l,l])+np.sum(Arr[0,1:l])+np.sum(Arr[l,1:l])
    middle = np.sum(Arr[1:l,1:l])
    tot = (middle*4+edges*2+corners)/4*8.4/(Resolution-1)**2
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
    tot = (middle*8+faces*4+edges*2+corners)/8*58.8/(Resolution-1)**3
    return tot

Resolution = 150

#Loads and sums all patient specific probability fields
#PDF = np.zeros([Resolution, Resolution, Resolution])
#for subdir, dirs, files in os.walk('C:\WinPython-64bit-3.5.4.1Qt5\Glucose\Patients'):
#    for file in files:       
#        PDF = PDF + np.load(subdir + "\\" + file)

#Loads pre-summed probability fields
PDF_2D = np.load('PDF_2D_smooth.npy')
PDF_3D = np.load('PDF_3D.npy')

print('Percentage Error in 2D is ' + str((Trap2D(PDF_2D)-62589)/62589*100) + '%')
print('Percentage Error in 3D is ' + str((Trap3D(PDF_3D)-62589)/62589*100) + '%')

#Dimensions of variables
SIt = np.linspace(-8.5, -1.5, Resolution)
Gt = np.linspace(0.2, 1.4, Resolution)
SIt1 = np.linspace(-8.5, -1.5, Resolution)

Conf_Int = [0, 0]
Conf_Int[0] = np.zeros([150,150])
Conf_Int[1] = np.zeros([150,150])
SIt_Med = np.zeros([150,150])
for i in range(150):
    for j in range(150):
        if np.sum(PDF_3D[j,i,:]) != 0:
            linear_prob = Trap1DCum(PDF_3D[j,i,:], SIt1)/Trap1D(PDF_3D[j,i,:], SIt1)
            Conf_Int[0][j,i] = SIt1[np.argmin(abs(linear_prob - 0.05))]
            Conf_Int[1][j,i] = SIt1[np.argmin(abs(linear_prob - 0.95))]
            SIt_Med[j,i] = SIt1[np.argmin(abs(linear_prob - 0.5))]
Conf_Width = 10**Conf_Int[1] - 10**Conf_Int[0]
Conf_Exp_Val = Trap2D(Conf_Width*PDF_2D/62589)
SIt1_Exp_Val = Trap2D(10**SIt_Med*PDF_2D/62589)
Conf_Up_Val = Trap2D(10**Conf_Int[1]*PDF_2D/62589)
Conf_Low_Val = Trap2D(10**Conf_Int[0]*PDF_2D/62589)
print('Expected Confidence Interval width is ' + str(Conf_Exp_Val))
print('Expected Value of SI(t+1) is ' + str(SIt1_Exp_Val) + ' with lower bound ' + str(Conf_Low_Val) + ' and upper bound ' + str(Conf_Up_Val))