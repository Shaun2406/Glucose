# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import scipy.integrate
plt.close("all")

Patient = scipy.io.loadmat('SP5005-1.mat')
Dia = 0

RawSI = [];
P = [];
PN = [];
for i in range(39):
  RawSI = np.append(RawSI, np.ones(60)*Patient['rawSI'][i,1])
  P = np.append(P, np.ones(60)*Patient['P'][i,1])
  PN = np.append(PN, np.ones(60)*Patient['PN'][i,1])

Uex = []
for i in range(69):
  Uex = np.append(Uex, np.ones(Patient['u'][i+1,0]-Patient['u'][i,0])*Patient['u'][i,1])
  
Idot = []
Gdot = []
Qdot = []
Uen = []
G = np.array([Patient['GIQ'][0,0]])
I = np.array([Patient['GIQ'][0,1]])
Q = np.array([Patient['GIQ'][0,2]])
P1 = np.array([0])
P2 = np.array([0])

for i in range(2340):
  Uen = np.append(Uen, min(max(Patient['uenmin'],Patient['k1'][0,Dia]*G[i]+Patient['k2'][0,Dia]),Patient['uenmax']))
  Gdot = np.append(Gdot, -Patient['pG']*G[i]-RawSI[i]*G[i]*Q[i]/(1+Patient['alpha_G']*Q[i])+(min(Patient['d2']*P2[i],Patient['Pmax'])+Patient['EGP'][0,1]-Patient['CNS']+PN[i])/Patient['Vg'])
  Idot = np.append(Idot, -Patient['nL']*I[i]/(1+Patient['alpha_I']*I[i])-Patient['nK']*I[i]-(I[i]-Q[i])*Patient['nI']+Uex[i]/Patient['Vi']+(1-Patient['xl'])*Uen[i]/Patient['Vi'])
  Qdot = np.append(Qdot, (I[i]-Q[i])*Patient['nI']-Patient['nC']*Q[i]/(1+Patient['alpha_G']*Q[i]))
  I = np.append(I,I[i]+Idot[i])
  G = np.append(G,G[i]+Gdot[i])
  Q = np.append(Q,Q[i]+Qdot[i])
  P1 = np.append(P1, P1[i]-Patient['d1']*P1[i]+P[i])
  P2 = np.append(P2, P2[i]-min(Patient['d2']*P2[i],Patient['Pmax'])+Patient['d1']*P1[i])

'''plt.figure()
plt.plot(range(2340),np.transpose(Gdot))'''
plt.figure()
plt.plot(range(2341),np.transpose(Patient['GIQ'][:,0]))
plt.plot(range(2341),np.transpose(G))
plt.title('Glucose')
'''plt.figure()
plt.plot(range(2340),np.transpose(Idot))'''
plt.figure()
plt.plot(range(2341),np.transpose(Patient['GIQ'][:,1]))
plt.plot(range(2341),np.transpose(I))
plt.title('Insulin - Plasma')
plt.figure()
plt.plot(range(2341),np.transpose(Patient['GIQ'][:,2]))
plt.plot(range(2341),np.transpose(Q))
plt.title('Insulin - Interstitial')