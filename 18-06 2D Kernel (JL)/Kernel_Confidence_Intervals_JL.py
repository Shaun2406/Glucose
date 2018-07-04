# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 12:06:41 2018

Confidence interval and expected values for JL kernel method
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.integrate
plt.close("all")

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
            linear_prob = scipy.integrate.cumtrapz(PDF_2D[:,i], SIt)/np.trapz(PDF_2D[:,i], SIt)
            Conf_Int[0][i] = SIt1[np.argmin(abs(linear_prob - 0.05))+1]
            Conf_Int[1][i] = SIt1[np.argmin(abs(linear_prob - 0.95))+1]
            SIt_Med[i] = SIt1[np.argmin(abs(linear_prob - 0.5))+1]
Conf_Width = Conf_Int[1] - Conf_Int[0]
Conf_Exp_Val = np.trapz(Conf_Width*PDF_1D/62589, SIt)
SIt1_Exp_Val = np.trapz(SIt_Med*PDF_1D/62589, SIt)
Conf_Up_Val = np.trapz(Conf_Int[1]*PDF_1D/62589, SIt)
Conf_Low_Val = np.trapz(Conf_Int[0]*PDF_1D/62589, SIt)
print('Expected Confidence Interval width is ' + str(Conf_Exp_Val))
print('Expected Value of SI(t+1) is ' + str(SIt1_Exp_Val) + ' with lower bound ' + str(Conf_Low_Val) + ' and upper bound ' + str(Conf_Up_Val))

np.save('Conf_Width_JL', Conf_Width)