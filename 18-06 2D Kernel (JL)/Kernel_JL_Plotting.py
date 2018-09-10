# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 12:06:41 2018

Plotting of PDF surface for JL kernel method
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
plt.close("all")

def Trap2D(Arr, xArr, yArr):
    tot = 0
    for i in range(len(xArr)-1):
        for j in range(len(yArr)-1):
            tot = tot+(Arr[j,i]+Arr[j+1,i+1]+Arr[j+1,i]+Arr[j,i+1])/4*(xArr[i+1]-xArr[i])*(yArr[j+1]-yArr[j])
    return tot

Resolution = 150

GlucData = pd.read_csv('C:\\WinPython-64bit-3.5.4.1Qt5\\Glucose\\GlucDataOverall.csv')
GlucData = GlucData[GlucData['SIt'] > 0]
GlucData = GlucData[GlucData['SIt+1'] > 0]   
GlucData = GlucData.reset_index()

#Loads and sums all patient specific probability fields
#PDF = np.zeros([Resolution, Resolution, Resolution])
#for subdir, dirs, files in os.walk('C:\WinPython-64bit-3.5.4.1Qt5\Glucose\Patients'):
#    for file in files:       
#        PDF = PDF + np.load(subdir + "\\" + file)

#Dimensions of variables
SIt = np.logspace(-8.5, -1.5, Resolution)
SIt1 = np.logspace(-8.5, -1.5, Resolution)

#Loads pre-summed probability fields
PDF_2D = np.load('PDF_2D_JL.npy')
print('Percentage Error in 2D is ' + str((Trap2D(PDF_2D, SIt, SIt1)-62589)/62589*100) + ' %')

#Surface Plot, 2D Field
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
SIt2D, SIt12D = np.meshgrid(SIt, SIt1)
ax.plot_surface(SIt2D, SIt12D, PDF_2D)
ax.set_xlabel('Sensitivity, SI(t)')
plt.ylabel('Sensitivity, SI(t+1)')

#Contour Plot, 2D Field
plt.figure()
plt.contour(SIt, SIt1, np.log(PDF_2D), 100)
plt.xlabel('Sensitivity, SI(t)')
plt.ylabel('Sensitivity, SI(t+1)')