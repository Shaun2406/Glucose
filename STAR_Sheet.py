# -*- coding: utf-8 -*-
"""
Created on Thu May 17 14:31:26 2018

@author: smd118
"""

"""Importing Libraries"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io
import scipy.integrate
import scipy.optimize
plt.close("all")

"""Loading Patient Data and Assigning Options"""
PatList = pd.read_csv('C:\WinPython-64bit-3.5.4.1Qt5\Glucose\PatientStructs\Patient_summary.csv')
PatList = PatList[PatList['file'] != '9aeed434-b0c9-4f8f-ad84-9683f900e7f1.GUIData']
failed = 0
total = 0
for i in PatList['file']:
    total += 1
    try:
        Patient = scipy.io.loadmat('C:\WinPython-64bit-3.5.4.1Qt5\Glucose\PatientStructs\\' + i[:-8])
        Patient = Patient['PatientStruct']
        Greal = np.array(Patient['Greal'][0,0])
        Treal = np.array(Patient['Treal'][0,0])
        GoalFeed = np.array(Patient['GF'][0,0])
        u = np.column_stack((np.array(Patient['u'][0,0][0,0]), np.array(Patient['u'][0,0][0,1])))
        P = np.column_stack((np.array(Patient['P'][0,0][0,0]), np.array(Patient['P'][0,0][0,1])))
        q = -1
        if len(P) > len(Treal):
            Pold = P
            for j in range(len(Treal)-1):
                if P[j+1,0] != Treal[j+1,0]:
                    A = (P[j+1,0]-P[j,0])*P[j,1]+(P[j+2,0]-P[j+1,0])*P[j+1,1]
                    P[j,1] = A
                    P = np.delete(P,j+1,0)
    except:
        failed += 1

with open("name.txt", "w") as output:
    output.write("asdf")

print("total %d failed %d" % (total, failed))