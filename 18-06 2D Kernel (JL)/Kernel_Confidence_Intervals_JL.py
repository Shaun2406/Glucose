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

def Trap2D(Arr, xArr, yArr):
    tot = 0
    for i in range(len(xArr)-1):
        for j in range(len(yArr)-1):
            tot = tot+(Arr[j,i]+Arr[j+1,i+1]+Arr[j+1,i]+Arr[j,i+1])/4*(xArr[i+1]-xArr[i])*(yArr[j+1]-yArr[j])
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

#Dimensions of variables
SIt = np.logspace(-8.5, -1.5, Resolution)
SIt1 = np.logspace(-8.5, -1.5, Resolution)

print('Percentage Error in 2D is ' + str((Trap2D(PDF_2D, SIt, SIt1)-62589)/62589*100) + '%')

Conf_Int = [0, 0]
Conf_Int[0] = np.zeros([150])
Conf_Int[1] = np.zeros([150])
SIt_Med = np.zeros([150])
for i in range(150):
        if np.sum(PDF_2D[:,i]) != 0:
            linear_prob = Trap1DCum(PDF_2D[:,i], SIt)/Trap1D(PDF_2D[:,i], SIt)
            Conf_Int[0][i] = SIt1[np.argmin(abs(linear_prob - 0.05))]
            Conf_Int[1][i] = SIt1[np.argmin(abs(linear_prob - 0.95))]
            SIt_Med[i] = SIt1[np.argmin(abs(linear_prob - 0.5))]
Conf_Width = Conf_Int[1] - Conf_Int[0]
Conf_Exp_Val = Trap1D(Conf_Width*PDF_1D/62589, SIt)
SIt1_Exp_Val = Trap1D(SIt_Med*PDF_1D/62589, SIt)
Conf_Up_Val = Trap1D(Conf_Int[1]*PDF_1D/62589, SIt)
Conf_Low_Val = Trap1D(Conf_Int[0]*PDF_1D/62589, SIt)
print('Expected Confidence Interval width is ' + str(Conf_Exp_Val))
print('Expected Value of SI(t+1) is ' + str(SIt1_Exp_Val) + ' with lower bound ' + str(Conf_Low_Val) + ' and upper bound ' + str(Conf_Up_Val))