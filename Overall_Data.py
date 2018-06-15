# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 12:16:04 2018

@author: smd118
"""

import pandas as pd

G1 = pd.read_csv('GlucDataGYULA.csv')
G2 = pd.read_csv('GlucDataSPRINT.csv')
G3 = pd.read_csv('GlucDataSTAR.csv')
G4 = pd.read_csv('GlucDataSTAR2015.csv')

GlucData = G1.append(G2)
GlucData = GlucData.append(G3)
GlucData = GlucData.append(G4)

GlucData.to_csv('C:\WinPython-64bit-3.5.4.1Qt5\Glucose\GlucDataOverall.csv')