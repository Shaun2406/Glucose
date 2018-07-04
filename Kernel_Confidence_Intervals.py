# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 12:06:41 2018

@author: smd118
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
import scipy.integrate
plt.close("all")

def Interpolate(idx, tgt):
    if linear_prob[idx] < tgt:
        inter = (tgt - linear_prob[idx])/(linear_prob[idx+1]-linear_prob[idx])*SIt1[idx+2] + (linear_prob[idx+1] - tgt)/(linear_prob[idx+1]-linear_prob[idx])*SIt1[idx+1]
    else:
        inter = (linear_prob[idx] - tgt)/(linear_prob[idx]-linear_prob[idx-1])*SIt1[idx] + (tgt - linear_prob[idx-1])/(linear_prob[idx]-linear_prob[idx-1])*SIt1[idx+1]
    return inter

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
Conf_Int_JL = np.load('Conf_Int_JL.npy')
Conf_Width_JL = np.load('Conf_Width_JL.npy')
PDF_1D = np.load('PDF_1D_JL.npy')


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
            linear_prob = scipy.integrate.cumtrapz(PDF_3D[j,i,:], SIt1)/np.trapz(PDF_3D[j,i,:], SIt1)
            Conf_Int[0][j,i] = Interpolate(np.argmin(abs(linear_prob - 0.05)), 0.05)
            Conf_Int[1][j,i] = Interpolate(np.argmin(abs(linear_prob - 0.95)), 0.95)
            SIt_Med[j,i] = Interpolate(np.argmin(abs(linear_prob - 0.5)), 0.5)
Conf_Width = 10**Conf_Int[1] - 10**Conf_Int[0]
Conf_Exp_Val = Trap2D(Conf_Width*PDF_2D/62589)
SIt1_Exp_Val = Trap2D(10**SIt_Med*PDF_2D/62589)
Conf_Up_Val = Trap2D(10**Conf_Int[1]*PDF_2D/62589)
Conf_Low_Val = Trap2D(10**Conf_Int[0]*PDF_2D/62589)
print('Expected Confidence Interval width is ' + str(Conf_Exp_Val))
print('Expected Value of SI(t+1) is ' + str(SIt1_Exp_Val) + ' with lower bound ' + str(Conf_Low_Val) + ' and upper bound ' + str(Conf_Up_Val))
print('')
JL_CI = 0.000413051333416
JL_LB = 0.00022801714853
JL_UB = 0.000641068481946
JL_EV = 0.000432574234546
ExpVal = 0.000426998692708

print('Expected Value of SI(t+1) is ' + str(((JL_EV - ExpVal) - (SIt1_Exp_Val - ExpVal))/(JL_EV - ExpVal)*100) + '% closer')
print('Confidence Interval is ' + str((JL_CI - Conf_Exp_Val)/(JL_CI)*100) + '% narrower')
print('Upper Bound is ' + str((JL_UB - Conf_Up_Val)/(JL_UB)*100) + '% lower')
print('Lower Bound is ' + str((Conf_Low_Val - JL_LB)/(JL_LB)*100) + '% higher')

for i in range(150):
    if PDF_1D[i] < 10 :
        Conf_Int_JL[1][i] = np.nan
    for j in range(150):
        if PDF_2D[j,i] < 10:
            Conf_Int[1][j,i] = np.nan

IWINMASK = Conf_Width < Conf_Width_JL*np.ones([150,150])
IWINMASK = IWINMASK*Conf_Width*PDF_2D/Trap2D(IWINMASK*PDF_2D)
print(Trap2D(IWINMASK))

ILOSEMASK = Conf_Width < Conf_Width_JL*np.ones([150,150])
ILOSEMASK = ILOSEMASK*Conf_Width_JL*np.ones([150,150])*PDF_2D/Trap2D(ILOSEMASK*PDF_2D)
print(Trap2D(ILOSEMASK))

IWIN = Conf_Width < Conf_Width_JL*np.ones([150,150])
IWIN = IWIN*PDF_2D
IWIN = Trap2D(IWIN)

ILOSE = Conf_Width > Conf_Width_JL*np.ones([150,150])
ILOSE = ILOSE*PDF_2D
ILOSE = Trap2D(ILOSE)

print(IWIN)
print(ILOSE)



fig = plt.figure()
SIt2D, Gt2D = np.meshgrid(SIt, Gt)
plt.contour(SIt2D, Gt2D, Conf_Width < Conf_Width_JL*np.ones([150,150]))

fig = plt.figure()
SIt2D, Gt2D = np.meshgrid(SIt, Gt)
plt.contour(SIt2D, Gt2D, Conf_Width > Conf_Width_JL*np.ones([150,150]))
#ax.plot_surface(SIt2D, Gt2D, , color = 'b')
#ax.set_xlabel('Sensitivity, SI(t)')
#ax.set_ylabel('Glucose, G(t)')