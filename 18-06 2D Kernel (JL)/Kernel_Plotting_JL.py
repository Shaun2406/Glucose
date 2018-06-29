# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 12:06:41 2018

@author: smd118
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from mpl_toolkits.mplot3d import Axes3D
import os
plt.close("all")

def Trap2D(Arr):
    l = len(Arr)-1
    corners = Arr[0,0]+Arr[l,l]+Arr[0,l]+Arr[l,0]
    edges = np.sum(Arr[1:l,0])+np.sum(Arr[1:l,l])+np.sum(Arr[0,1:l])+np.sum(Arr[l,1:l])
    middle = np.sum(Arr[1:l,1:l])
    tot = (middle*4+edges*2+corners)/4*(10**-1.8 - 10**-8.5)**2/(Resolution-1)**2
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

GlucData = pd.read_csv('C:\\WinPython-64bit-3.5.4.1Qt5\\Glucose\\GlucDataOverall.csv')
GlucData = GlucData[GlucData['SIt'] > 0]
GlucData = GlucData[GlucData['SIt+1'] > 0]   
GlucData = GlucData.reset_index()

print(np.max(GlucData['SIt+1']))
print(np.max(GlucData['SIt']))
print(10**-1.5)
#Loads and sums all patient specific probability fields
#PDF = np.zeros([Resolution, Resolution, Resolution])
#for subdir, dirs, files in os.walk('C:\WinPython-64bit-3.5.4.1Qt5\Glucose\Patients'):
#    for file in files:       
#        PDF = PDF + np.load(subdir + "\\" + file)

#Loads pre-summed probability fields
PDF_2D = np.load('PDF_2D_JL.npy')

print('Percentage Error in 2D is ' + str((Trap2D(PDF_2D)-62589)/62589*100) + ' %')

#Dimensions of variables
SIt = np.linspace(10**-8.5, 10**-1.8, Resolution)
SIt1 = np.linspace(10**-8.5, 10**-1.8, Resolution)

#Surface Plot, 2D Field
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
SIt2D, SIt12D = np.meshgrid(SIt, SIt1)
ax.plot_surface(SIt2D, SIt12D, PDF_2D)
ax.set_xlabel('Sensitivity, SI(t)')
plt.ylabel('Sensitivity, SI(t+1)')

#Contour Plot, 2D Field
plt.figure()
plt.contour(SIt, SIt1, PDF_2D, 100)
plt.xlabel('Sensitivity, SI(t)')
plt.ylabel('Sensitivity, SI(t+1)')