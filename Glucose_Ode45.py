# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import scipy.integrate

plt.close("all")

Patient = scipy.io.loadmat('SP5005-1.mat')
Dia = 0

RawSI = np.zeros([2341])
P = np.zeros([2341])
PN = np.zeros([2341])
for i in range(39):
  RawSI[i*60:i*60+60] = np.ones(60)*Patient['rawSI'][i,1]
  P[i*60:i*60+60] = np.ones(60)*Patient['P'][i,1]
  PN[i*60:i*60+60] = np.ones(60)*Patient['PN'][i,1]
RawSI[2340] = RawSI[2339] 
P[2340] = P[2339] 
PN[2340] = PN[2339] 

Uex = np.zeros([2341]);
for i in range(69):
  Uex[Patient['u'][i,0]:Patient['u'][i+1,0]] = np.ones(Patient['u'][i+1,0]-Patient['u'][i,0])*Patient['u'][i,1]

def ICING2(t, G):
  Uent = min(max(Patient['uenmin'],Patient['k1'][0,Dia]*G[0]+Patient['k2'][0,Dia]),Patient['uenmax'])
  RawSIt = RawSI[int(t)]
  Pt = P[math.floor(t)]
  PNt = PN[math.floor(t)] 
  Uext = Uex[math.floor(t)]
  Gdot = np.zeros(5)
  Gdot[0] = -Patient['pG']*G[0]-RawSIt*G[0]*G[2]/(1+Patient['alpha_G']*G[2])+(min(Patient['d2']*G[4],Patient['Pmax'])+Patient['EGP'][0,1]-Patient['CNS']+PNt)/Patient['Vg']
  Gdot[1] = -Patient['nL']*G[1]/(1+Patient['alpha_I']*G[1])-Patient['nK']*G[1]-(G[1]-G[2])*Patient['nI']+Uext/Patient['Vi']+(1-Patient['xl'])*Uent/Patient['Vi']
  Gdot[2] = (G[1]-G[2])*Patient['nI']-Patient['nC']*G[2]/(1+Patient['alpha_G']*G[2])
  Gdot[3] = -Patient['d1']*G[3]+Pt
  Gdot[4] = -min(Patient['d2']*G[4],Patient['Pmax'])+Patient['d1']*G[3]
  return Gdot

t = range(2341)  
y = np.zeros([2341,5])
y[0,:] = np.append(Patient['GIQ'][0,:], np.zeros(2))
r = scipy.integrate.ode(ICING2).set_integrator("dopri5")
for i in range(1,2341):
  r.set_initial_value(y[i-1,:],i-1)
  y[i,:] = r.integrate(t[i])

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