# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 15:48:43 2018

@author: Shaun
"""

import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
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
  
test_length = 62589

q = -1

Conf_Width = np.zeros(test_length)
Conf_Int = [0, 0, 0]
Conf_Int[0] = np.zeros(test_length)
Conf_Int[1] = np.zeros(test_length)
Conf_Int[2] = np.zeros(test_length)
test_all = np.zeros([0,3])
for j in range(5):
    
    res = 300
    
    grid_pts = np.linspace(-8.5, -1.5, res)
    conf_ints = np.load('..\\CV Results\\PDF_3D_' + str(j+1) + '.npy')
    test_pts = np.load('..\\CV Results\\TEST_3D_' + str(j+1) + '.npy')
    test_all = np.append(test_all, test_pts, axis = 0)
    for i in range(len(test_pts)):
        q = q+1
        if np.trapz(conf_ints[i,:], grid_pts) != 0:
            linear_prob = scipy.integrate.cumtrapz(conf_ints[i,:], grid_pts)/np.trapz(conf_ints[i,:], grid_pts)
            Conf_Int[0][q] = Interpolate(np.argmin(abs(linear_prob - 0.05)), 0.05)
            Conf_Int[1][q] = Interpolate(np.argmin(abs(linear_prob - 0.95)), 0.95)
            Conf_Int[2][q] = Interpolate(np.argmin(abs(linear_prob - 0.50)), 0.50)
            Conf_Width[q] = 10**Conf_Int[1][q]-10**Conf_Int[0][q]
        else:
            Conf_Int[0][q] = 0
            Conf_Int[1][q] = 1
            Conf_Width[q] = 1
q = -1

Conf_Width_JL = np.zeros(test_length)
Conf_Int_JL = [0, 0, 0]
Conf_Int_JL[0] = np.zeros(test_length)
Conf_Int_JL[1] = np.zeros(test_length)
Conf_Int_JL[2] = np.zeros(test_length)

for j in range(5):
    
    res = 500
    
    grid_pts = np.linspace(0, 0.016, res)
    conf_ints = np.load('..\\18-06 2D Kernel (JL)\\PDF_JL_' + str(j+1) + '.npy')
    test_pts = np.load('..\\18-06 2D Kernel (JL)\\TEST_JL_' + str(j+1) + '.npy')
    
    for i in range(len(test_pts)):
        q = q+1
        if np.trapz(conf_ints[i,:], grid_pts) != 0:
            linear_prob = scipy.integrate.cumtrapz(conf_ints[i,:], grid_pts)/np.trapz(conf_ints[i,:], grid_pts)
            Conf_Int_JL[0][q] = Interpolate(np.argmin(abs(linear_prob - 0.05)), 0.05)
            Conf_Int_JL[1][q] = Interpolate(np.argmin(abs(linear_prob - 0.95)), 0.95)
            Conf_Int_JL[2][q] = Interpolate(np.argmin(abs(linear_prob - 0.50)), 0.50)
            Conf_Width_JL[q] = Conf_Int_JL[1][q]-Conf_Int_JL[0][q]
        else:
            Conf_Int_JL[0][q] = 0
            Conf_Int_JL[1][q] = 1
            Conf_Width_JL[q] = 1
q = -1

Conf_Out = np.zeros(62577)
Conf_Out_JL = np.zeros(62577)
Conf_I = [0, 0, 0]
Conf_I[0] = np.zeros(62577)
Conf_I[1] = np.zeros(62577)
Conf_I[2] = np.zeros(62577)
Conf_I_JL = [0, 0, 0]
Conf_I_JL[0] = np.zeros(62577)
Conf_I_JL[1] = np.zeros(62577)
Conf_I_JL[2] = np.zeros(62577)
test_used = np.zeros([62577, 3])
for j in range(test_length):
    if Conf_Width_JL[j] != 1 and Conf_Width[j] != 1:
        q = q+1
        Conf_Out[q] = Conf_Width[j]
        Conf_Out_JL[q] = Conf_Width_JL[j]
        Conf_I[0][q] = 10**Conf_Int[0][j]
        Conf_I[1][q] = 10**Conf_Int[1][j]
        Conf_I[2][q] = 10**Conf_Int[2][j]
        Conf_I_JL[0][q] = Conf_Int_JL[0][j]
        Conf_I_JL[1][q] = Conf_Int_JL[1][j]
        Conf_I_JL[2][q] = Conf_Int_JL[2][j]
        test_used[q,:] = test_all[j,:]
narrower = Conf_Out < Conf_Out_JL
narrow_3D = np.sum(Conf_Out*narrower)/np.sum(narrower)
narrow_2D = np.sum(Conf_Out_JL*narrower)/np.sum(narrower)
wider = Conf_Out > Conf_Out_JL
wide_3D = np.sum(Conf_Out*wider)/np.sum(wider)
wide_2D = np.sum(Conf_Out_JL*wider)/np.sum(wider)    
print('Overall Results')
print('3D kernel method is narrower in ' + str("%.2f" % (np.sum(narrower)/test_length*100)) + '% of cases')
print('In these cases, it is ' + str("%.2f" % ((narrow_2D-narrow_3D)/narrow_2D*100)) + '% narrower on average')    
print('3D kernel method is wider in ' + str("%.2f" % (np.sum(wider)/test_length*100)) + '% of cases')
print('In these cases, it is ' + str("%.2f" % ((wide_3D-wide_2D)/wide_2D*100)) + '% wider on average')    
print('')
print('On average, the 3D method is ' + str("%.2f" %((np.mean(Conf_Out_JL)-np.mean(Conf_Out))/np.mean(Conf_Out_JL)*100)) + '% narrower')
print('On average, the upper bound is ' + str("%.2f" %((np.mean(Conf_I_JL[1])-np.mean(Conf_I[1]))/np.mean(Conf_I_JL[1])*100)) + '% lower')
print('On average, the lower bound is ' + str("%.2f" %((np.mean(Conf_I[0])-np.mean(Conf_I_JL[0]))/np.mean(Conf_I_JL[0])*100)) + '% higher')
print('')
print('3D 5% boundary is ' + str("%.6f" %(np.mean(Conf_I[0]))) + ' and 95% boundary ' + str("%.6f" %(np.mean(Conf_I[1]))))
print('2D 5% boundary is ' + str("%.6f" %(np.mean(Conf_I_JL[0]))) + ' and 95% boundary ' + str("%.6f" %(np.mean(Conf_I_JL[1]))))
print('3D 90% width is ' + str("%.6f" %(np.mean(Conf_Out))) + ' and 2D 90% width ' + str("%.6f" %(np.mean(Conf_Out_JL))))
print('3D expected value is ' + str("%.6f" %(np.mean(Conf_I[2]))) + ' and 2D expected value ' + str("%.6f" %(np.mean(Conf_I_JL[2]))))
#plt.figure()
#plt.contour(grid_pts[1], grid_pts[0], narrower)
#plt.plot(np.sort(Conf_Out_JL),'r-')

Gt_fail = test_used[:,1]*wider
SI_fail = test_used[:,0]*wider
Gt_fail = Gt_fail[Gt_fail != 0]
SI_fail = SI_fail[SI_fail != 0]

print('Mean Gt is ' + str("%.6f" %(np.mean(10**test_used[:,1]))) + ' and is '  + str("%.6f" %(np.mean(10**Gt_fail))) + ' when wider')
print('Mean SI is ' + str("%.6f" %(np.mean(10**test_used[:,0]))) + ' and is '  + str("%.6f" %(np.mean(10**SI_fail))) + ' when wider')