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

print('Percentage Error in 2D is ' + str((Trap2D(PDF_2D)-62589)/62589*100) + ' %')
print('Percentage Error in 3D is ' + str((Trap3D(PDF_3D)-62589)/62589*100) + ' %')

#Dimensions of variables
SIt = np.linspace(-8.5, -1.5, Resolution)
Gt = np.linspace(0.2, 1.4, Resolution)
SIt1 = np.linspace(-8.5, -1.5, Resolution)

#Surface Plot, 2D Field
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
SIt2D, Gt2D = np.meshgrid(SIt, Gt)
ax.plot_surface(SIt2D, Gt2D, PDF_2D)
ax.set_xlabel('Sensitivity, SI(t)')
ax.set_ylabel('Glucose, G(t)')

#Contour Plot, 2D Field
plt.figure()
plt.contour(SIt, Gt, PDF_2D, 100)
plt.xlabel('Sensitivity, SI(t)')
plt.ylabel('Glucose, G(t)')

#Video Writer
'''FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Contour_Plot_3D', artist='Shaun Davidson',
                comment='3D Kernel Field Plot')
writer = FFMpegWriter(fps=15, metadata=metadata)

fig = plt.figure()
with writer.saving(fig, "3D_Contours_Normal.mp4", 150):
    for i in range(150):
        plt.contour(SIt, Gt, (PDF_3D[:,:,i]), 100)
        writer.grab_frame()
        plt.gcf().clear()'''

#Plots Projections of the 3D Kernel Field onto 2D Planes        
plt.figure()
plt.contour(Gt, SIt1, np.sum(PDF_3D, 0), 100)
plt.xlabel('Glucose, G(t)')
plt.ylabel('Sensitivity, SI(t+1)')       

plt.figure()
plt.contour(SIt, SIt1, np.sum(PDF_3D, 1), 100)
plt.xlabel('Sensitivity, SI(t)')
plt.ylabel('Sensitivity, SI(t+1)')  

plt.figure()
plt.contour(SIt, Gt, np.sum(PDF_3D, 2), 100)
plt.xlabel('Sensitivity, SI(t)')
plt.ylabel('Glucose, G(t)')  