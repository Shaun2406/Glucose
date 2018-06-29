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

def Trap1D(Arr):
    l = len(Arr)-1
    edges = Arr[0]+Arr[l]
    middle = np.sum(Arr[1:l])
    tot = (middle*2+edges)/2*(10**-1.8 - 10**-8.5)/(Resolution-1)
    return tot

def Trap2D(Arr):
    l = len(Arr)-1
    corners = Arr[0,0]+Arr[l,l]+Arr[0,l]+Arr[l,0]
    edges = np.sum(Arr[1:l,0])+np.sum(Arr[1:l,l])+np.sum(Arr[0,1:l])+np.sum(Arr[l,1:l])
    middle = np.sum(Arr[1:l,1:l])
    tot = (middle*4+edges*2+corners)/4*(10**-1.8 - 10**-8.5)**2/(Resolution-1)**2
    return tot

Resolution = 150

#Loads and sums all patient specific probability fields
#PDF = np.zeros([Resolution, Resolution, Resolution])
#for subdir, dirs, files in os.walk('C:\WinPython-64bit-3.5.4.1Qt5\Glucose\Patients'):
#    for file in files:       
#        PDF = PDF + np.load(subdir + "\\" + file)

#Loads pre-summed probability fields
PDF_2D = np.load('PDF_2D_JL.npy')
PDF_1D = np.load('PDF_1D_JL.npy')
print('Percentage Error in 2D is ' + str((Trap2D(PDF_2D)-62589)/62589*100) + '%')

#Dimensions of variables
SIt = np.linspace(10**-8.5, 10**-1.8, Resolution)
SIt1 = np.linspace(10**-8.5, 10**-1.8, Resolution)

Conf_Int = [0, 0]
Conf_Int[0] = np.zeros([150])
Conf_Int[1] = np.zeros([150])
SIt_Med = np.zeros([150])
for i in range(150):
        if np.sum(PDF_2D[:,i]) != 0:
            linear_prob = np.cumsum(PDF_2D[:,i])/np.sum(PDF_2D[:,i])
            Conf_Int[0][i] = SIt1[np.argmin(abs(linear_prob - 0.05))]
            Conf_Int[1][i] = SIt1[np.argmin(abs(linear_prob - 0.95))]
            SIt_Med[i] = SIt1[np.argmin(abs(linear_prob - 0.5))]
Conf_Width = Conf_Int[1] - Conf_Int[0]
Conf_Exp_Val = Trap1D(Conf_Width*PDF_1D/62589)
SIt1_Exp_Val = Trap1D(SIt_Med*PDF_1D/62589)
Conf_Up_Val = Trap1D(Conf_Int[1]*PDF_1D/62589)
Conf_Low_Val = Trap1D(Conf_Int[0]*PDF_1D/62589)
print('Expected Confidence Interval width is ' + str(Conf_Exp_Val))
print('Expected Value of SI(t+1) is ' + str(SIt1_Exp_Val) + ' with lower bound ' + str(Conf_Low_Val) + ' and upper bound ' + str(Conf_Up_Val))