# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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
Patient = scipy.io.loadmat('SP5006-1.mat')
Dia = 0
Lm = len(Patient['GIQ'])
Lh = int(Lm/60)

"""Extending Array Lengths to be Consistent"""
P = np.zeros([Lm])
PN = np.zeros([Lm])
for i in range(Lh):
  P[i*60:i*60+60] = Patient['P'][i,1]
  PN[i*60:i*60+60] = Patient['PN'][i,1]
P[Lm-1] = P[Lm-2] 
PN[Lm-1] = PN[Lm-2] 
Uex = np.zeros([Lm]);
for i in range(len(Patient['u'])-1):
  Uex[Patient['u'][i,0]:Patient['u'][i+1,0]] = Patient['u'][i,1]
Ginter = np.interp(range(Lm),Patient['Treal'][:,0],Patient['Greal'][:,0])

"""ICING2 Function for Least-Squares Identification"""
def ICING2(t, G):
  Uent = min(max(Patient['uenmin'],Patient['k1'][0,Dia]*Ginter[int(t)]+Patient['k2'][0,Dia]),Patient['uenmax'])
  Pt = P[int(t)]
  Uext = Uex[int(t)]
  Gdot = np.zeros(6)
  Gdot[0] = Ginter[int(t)]*G[3]/(1+Patient['alpha_G']*G[3])
  Gdot[1] = -Patient['pG']*Ginter[int(t)]+(min(Patient['d2']*G[5],Patient['Pmax'])+Patient['EGP'][0,1]-Patient['CNS'])/Patient['Vg']
  Gdot[2] = -Patient['nL']*G[2]/(1+Patient['alpha_I']*G[2])-Patient['nK']*G[2]-(G[2]-G[3])*Patient['nI']+Uext/Patient['Vi']+(1-Patient['xl'])*Uent/Patient['Vi']
  Gdot[3] = (G[2]-G[3])*Patient['nI']-Patient['nC']*G[3]/(1+Patient['alpha_G']*G[3])
  Gdot[4] = -Patient['d1']*G[4]+Pt
  Gdot[5] = -min(Patient['d2']*G[5],Patient['Pmax'])+Patient['d1']*G[4]
  return Gdot

"""ICING2 Function for Integration"""
def ICING2int(t, G):
  Uent = min(max(Patient['uenmin'],Patient['k1'][0,Dia]*G[0]+Patient['k2'][0,Dia]),Patient['uenmax'])
  SIt = SI[int(t)]
  Pt = P[int(t)]
  PNt = PN[int(t)] 
  Uext = Uex[int(t)]
  Gdot = np.zeros(5)
  Gdot[0] = -Patient['pG']*G[0]-SIt*G[0]*G[2]/(1+Patient['alpha_G']*G[2])+(min(Patient['d2']*G[4],Patient['Pmax'])+Patient['EGP'][0,1]-Patient['CNS']+PNt)/Patient['Vg']
  Gdot[1] = -Patient['nL']*G[1]/(1+Patient['alpha_I']*G[1])-Patient['nK']*G[1]-(G[1]-G[2])*Patient['nI']+Uext/Patient['Vi']+(1-Patient['xl'])*Uent/Patient['Vi']
  Gdot[2] = (G[1]-G[2])*Patient['nI']-Patient['nC']*G[2]/(1+Patient['alpha_G']*G[2])
  Gdot[3] = -Patient['d1']*G[3]+Pt
  Gdot[4] = -min(Patient['d2']*G[4],Patient['Pmax'])+Patient['d1']*G[3]
  return Gdot

"""Integrates the Least-Squares Function, Assuming a Linear Glucose Profile"""
t = range(Lm)  
y_id = np.zeros([Lm,6])
y_id[0,0] = Ginter[0]*Patient['GIQ'][0,2]/(1+Patient['alpha_G']*Patient['GIQ'][0,2])
y_id[0,1] = -(Ginter[60]-Ginter[0])-Patient['pG']*Ginter[0]+(min(Patient['d2']*0,Patient['Pmax'])+Patient['EGP'][0,1]-Patient['CNS'])/Patient['Vg']
y_id[0,2] = (Patient['Uo']+(1-Patient['xl'])*min(max(Patient['uenmin'],Patient['k1'][0,Dia]*Patient['Greal'][0]+Patient['k2'][0,Dia]),Patient['uenmax']))/Patient['Vi']/(Patient['nK']+Patient['nL']+0.5*Patient['nI'])
y_id[0,3] = y_id[0,2]/2
y_id[0,4] = P[0]/Patient['d1']
y_id[0,5] = P[0]/Patient['d2']
r = scipy.integrate.ode(ICING2).set_integrator("dopri5")
for i in range(1,Lm):
  r.set_initial_value(y_id[i-1,:],i-1)
  y_id[i,:] = r.integrate(t[i])
  
"""Solves for Hourly SI"""
RawSI = np.zeros([Lh])
for i in range(Lh):
  A = y_id[60*i+60,0] - y_id[60*i,0]
  B = -(Ginter[60*i+60]-Ginter[60*i])+y_id[60*i+60,1]-y_id[60*i,1]
  RawSI[i] = B/A

"""Extends SI and Integrates all Waveforms"""
SI = np.zeros([Lm])
for i in range(Lh):
  SI[i*60:i*60+60] = RawSI[i]
SI[Lm-1] = SI[Lm-2] 
y = np.zeros([Lm,5])
y[0,0] = Patient['Greal'][0]
y[0,1] = (Patient['Uo']+(1-Patient['xl'])*min(max(Patient['uenmin'],Patient['k1'][0,Dia]*Patient['Greal'][0]+Patient['k2'][0,Dia]),Patient['uenmax']))/Patient['Vi']/(Patient['nK']+Patient['nL']+0.5*Patient['nI'])
y[0,2] = y_id[0,2]/2
y[0,3] = P[0]/Patient['d1']
y[0,4] = P[0]/Patient['d2']
r = scipy.integrate.ode(ICING2int).set_integrator("dopri5")
for i in range(1,Lm):
  r.set_initial_value(y[i-1,:],i-1)
  y[i,:] = r.integrate(t[i])

"""Plots Figures"""
plt.figure()
plt.plot(t,np.transpose(Patient['GIQ'][:,0]))
plt.plot(t,y[:,0])
plt.plot(Patient['Treal'],Patient['Greal'],'kx')
plt.title('Glucose')
plt.xlabel('Time (mins)')
plt.ylabel('Glucose (mmol/L)')
plt.legend(['SWEET','Python','Measured'])

plt.figure()
plt.plot(t,np.transpose(Patient['GIQ'][:,1]))
plt.plot(t,y[:,1])
plt.title('Insulin - Plasma')
plt.xlabel('Time (mins)')
plt.ylabel('Insulin (mmol/L)')

plt.figure()
plt.plot(t,np.transpose(Patient['GIQ'][:,2]))
plt.plot(t,y[:,2])
plt.title('Insulin - Interstitial')
plt.xlabel('Time (mins)')
plt.ylabel('Insulin (mmol/L)')

plt.figure()
plt.plot(range(Lh), RawSI)
plt.plot(range(Lh), Patient['rawSI'][:,1])
plt.title('Insulin Sensitivity')
plt.xlabel('Time (mins)')
plt.ylabel('Sensitivity')
