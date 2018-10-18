# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 12:06:41 2018

@author: smd118

Outdated and not a whole lotta use sadly, use CI compare instead
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

def Trap2D(Arr):
    l = len(Arr)-1
    corners = Arr[0,0]+Arr[l,l]+Arr[0,l]+Arr[l,0]
    edges = np.sum(Arr[1:l,0])+np.sum(Arr[1:l,l])+np.sum(Arr[0,1:l])+np.sum(Arr[l,1:l])
    middle = np.sum(Arr[1:l,1:l])
    tot = (middle*4+edges*2+corners)/4*8.4/(res-1)**2
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

res = 300

#Loads pre-summed probability fields
PDF_2D = np.load('PDF_2D.npy')
PDF_3D = np.load('PDF_3D.npy')
Conf_Int_JL = np.load('JL_Conf_Int.npy')
Conf_Width_JL = np.load('JL_Conf_Width.npy')

print('Percentage Error in 2D is ' + str("%.4f" % ((Trap2D(PDF_2D)-62589)/62589*100)) + '%')
print('Percentage Error in 3D is ' + str("%.4f" % ((Trap3D(PDF_3D)-62589)/62589*100)) + '%')

#Dimensions of variables
Gt = np.linspace(0.2, 1.4, res)
grid_pts = np.linspace(-8.5, -1.5, res)
Conf_Int = []
for i in range(5):
    Conf_Int.append(np.zeros([res,res]))
for i in range(res):
    for j in range(res):
        if np.sum(PDF_3D[j,i,:]) != 0:
            linear_prob = scipy.integrate.cumtrapz(PDF_3D[j,i,:], grid_pts)/np.trapz(PDF_3D[j,i,:], grid_pts)
            Conf_Int[0][j,i] = Interpolate(np.argmin(abs(linear_prob - 0.05)), 0.05)
            Conf_Int[1][j,i] = Interpolate(np.argmin(abs(linear_prob - 0.50)), 0.50)
            Conf_Int[2][j,i] = Interpolate(np.argmin(abs(linear_prob - 0.95)), 0.95)
Conf_Width = 10**Conf_Int[2] - 10**Conf_Int[0]
Conf_Exp_Val = Trap2D(Conf_Width*PDF_2D/62589)
SIt1_Exp_Val = Trap2D(10**Conf_Int[1]*PDF_2D/62589)
Conf_Up_Val = Trap2D(10**Conf_Int[2]*PDF_2D/62589)
Conf_Low_Val = Trap2D(10**Conf_Int[0]*PDF_2D/62589)
print('Expected Confidence Interval width is ' + str("%.6f" % Conf_Exp_Val))
print('Expected Value of SI(t+1) is ' + str("%.6f" % SIt1_Exp_Val) + ' with lower bound ' + str("%.6f" % Conf_Low_Val) + ' and upper bound ' + str("%.6f" % Conf_Up_Val))
print('')
JL_CI = 0.000397966927518
JL_LB = 0.000229263573861
JL_UB = 0.000627230501379
JL_EV = 0.000425542112961
ExpVal = 0.000426998692708

print('Confidence Interval is ' + str("%.2f" % ((JL_CI - Conf_Exp_Val)/(JL_CI)*100)) + '% narrower')
print('Upper Bound is ' + str("%.2f" % ((JL_UB - Conf_Up_Val)/(JL_UB)*100)) + '% lower')
print('Lower Bound is ' + str("%.2f" % ((Conf_Low_Val - JL_LB)/(JL_LB)*100)) + '% higher')
print('')


Conf_Width_JL = np.transpose(Conf_Width_JL)*np.ones([500,500])
JL_Conf = scipy.interpolate.interp2d(np.linspace(0, 0.016, 500), np.linspace(10**0.2, 10**1.4, 500), Conf_Width_JL)
Conf_Width_JL = JL_Conf.__call__(10**grid_pts, 10**Gt)

#Where CI is narrower than existing method
#Narrower = Conf_Width < JL_Conf_Resample
#Narrower_Avr_CI = Trap2D(Narrower*Conf_Width*PDF_2D)/Trap2D(Narrower*PDF_2D)
#Narrower_Avr_JL = Trap2D(Narrower*JL_Conf_Resample*PDF_1D)/Trap2D(Narrower*PDF_1D)

#Wider = Conf_Width > JL_Conf_Resample
#Wider_Avr_CI = Trap2D(Wider*Conf_Width*PDF_2D)/Trap2D(Wider*PDF_2D)
#Wider_Avr_JL = Trap2D(Wider*JL_Conf_Resample*PDF_1D)/Trap2D(Wider*PDF_1D)

#print('CI is narrower in ' + str("%.2f" % Trap2D(Narrower*PDF_2D*100/62589)) + '% of cases')
#print('In these cases CI is ' + str("%.2f" % ((Narrower_Avr_JL-Narrower_Avr_CI)/Narrower_Avr_JL*100)) + '% narrower on average')

#print('CI is wider in ' + str("%.2f" % Trap2D(Wider*PDF_2D*100/62589)) + '% of cases')
#print('In these cases CI is ' + str("%.2f" % ((Wider_Avr_CI-Wider_Avr_JL)/Wider_Avr_JL*100)) + '% wider on average')

fig = plt.figure()
SIt2D, Gt2D = np.meshgrid(grid_pts, Gt)
plt.contour(SIt2D, Gt2D, Conf_Width < Conf_Width_JL)

fig = plt.figure()
SIt2D, Gt2D = np.meshgrid(grid_pts, Gt)
ax = fig.add_subplot(111, projection = '3d')
ax.plot_surface(SIt2D, Gt2D, Conf_Width_JL, color = 'b')
ax.set_xlabel('Sensitivity, SI(t)')
ax.set_ylabel('Glucose, G(t)')

#fig = plt.figure()
#SIt2D, Gt2D = np.meshgrid(grid_pts, Gt)
#ax = fig.add_subplot(111, projection = '3d')
ax.plot_surface(SIt2D, Gt2D, Conf_Width, color = 'r')
ax.set_xlabel('Sensitivity, SI(t)')
ax.set_ylabel('Glucose, G(t)')

np.savetxt("Conf_LB", 10**Conf_Int[0], delimiter=",")
np.savetxt("Conf_UB", 10**Conf_Int[2], delimiter=",")