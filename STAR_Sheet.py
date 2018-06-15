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
import datetime
import os
plt.close("all")

"""General ICING Patient Parameters"""
Pat = {'pG': 0.006, 'alphaG': 1/65, 'd1': -np.log(0.5)/20, 'd2': -np.log(0.5)/100, 'Pmax': 6.11, 'EGP': 1.16, 'CNS': 0.3, 'Vg': 13.3, 'nI': 0.006, 'nC': 0.006, 'alphaI': 0.0017, 'Vi': 4.0, 'xl': 0.67, 'nK': 0.0542, 'nL': 0.1578, 'uenmin': 16.7, 'uenmax': 266.7, 'k1': np.array([14.9, 0.0, 4.9]), 'k2': np.array([-49.9, 16.7, -27.4])}

"""Defining Functions"""
def resample(inpmat, t):
    t = t[:,0]
    expmat = np.zeros(t[-1]+1)
    for i in range(len(inpmat)-1):
        expmat[int(inpmat[i,0]):int(inpmat[i+1,0])] = inpmat[i,1]
    outmat = np.zeros(len(t)-1)
    for i in range(len(t)-1):
        outmat[i] = np.average(expmat[t[i]:t[i+1]])   
    return outmat

def expand(inpmat, t):
    expmat = np.zeros(t[-1].astype(int)+1)
    for i in range(len(inpmat)-1):
        expmat[int(inpmat[i,0]):int(inpmat[i+1,0])] = inpmat[i,1]
    return expmat

def ICING2(t, G):
  Uent = min(max(Pat['uenmin'],Pat['k1'][Dia]*Gnt[int(t)]+Pat['k2'][Dia]),Pat['uenmax'])
  Pt = Pmin[int(t)]
  ut = umin[int(t)]
  PNt = PNmin[int(t)]
  Gdot = np.zeros(6)
  Gdot[0] = Gnt[int(t)]*G[3]/(1+Pat['alphaG']*G[3])
  Gdot[1] = -Pat['pG']*Gnt[int(t)]+(min(Pat['d2']*G[5],Pat['Pmax'])+Pat['EGP']-Pat['CNS']+PNt)/Pat['Vg']
  Gdot[2] = -Pat['nL']*G[2]/(1+Pat['alphaI']*G[2])-Pat['nK']*G[2]-(G[2]-G[3])*Pat['nI']+ut/Pat['Vi']+(1-Pat['xl'])*Uent/Pat['Vi']
  Gdot[3] = (G[2]-G[3])*Pat['nI']-Pat['nC']*G[3]/(1+Pat['alphaG']*G[3])
  Gdot[4] = -Pat['d1']*G[4]+Pt
  Gdot[5] = -min(Pat['d2']*G[5],Pat['Pmax'])+Pat['d1']*G[4]
  return Gdot

"""Loading Patient Data and Iterates through it"""
Gluc = pd.DataFrame()
for subdir, dirs, files in os.walk('C:\WinPython-64bit-3.5.4.1Qt5\Glucose\PatientStructsSTAR'):
    for file in files:
        Patient = scipy.io.loadmat(subdir + "\\" + file)
        Patient = Patient['PatientStruct']
        '''Dia = Patient['DiabeticStatus'][0,0]'''
        '''Dia = 0'''
        '''Gender = Patient['Gender'][0,0]'''
        '''Operative = Patient['Operative'] == 'True' '''
        '''Operative = Patient['Apache2_Score'] > 10'''
        Prot = datetime.datetime(Patient['ProtocolStartTime'][0,0][0,0], Patient['ProtocolStartTime'][0,0][0,1], Patient['ProtocolStartTime'][0,0][0,2], Patient['ProtocolStartTime'][0,0][0,3], Patient['ProtocolStartTime'][0,0][0,4], Patient['ProtocolStartTime'][0,0][0,5])
        Admis = datetime.datetime(Patient['ICUAdminDate'][0,0][0,0], Patient['ICUAdminDate'][0,0][0,1], Patient['ICUAdminDate'][0,0][0,2], Patient['ICUAdminDate'][0,0][0,3], Patient['ICUAdminDate'][0,0][0,4], Patient['ICUAdminDate'][0,0][0,5])
        '''PreProt = Prot-Admis
        PreProt = PreProt.days*1440+PreProt.seconds/60
        
        if file == '08a4abcd-0b10-46a4-9918-d9c66d0afc30.mat':
            Greal = np.array(Patient['Greal'][0,0])
            Treal = np.array(Patient['Treal'][0,0][0])
            Treal = np.reshape(Treal,[7,1])
        else:
            Greal = np.array(Patient['Greal'][0,0])
            Treal = np.array(Patient['Treal'][0,0])
            
        Lm = Treal[-1]
        Lh = int(Lm/60)
        GoalFeed = np.column_stack((np.array(Patient['GF'][0,0][0,0]), np.array(Patient['GF'][0,0][0,1])))
        u = np.column_stack((np.array(Patient['u'][0,0][0,0]), np.array(Patient['u'][0,0][0,1])))
        P = np.column_stack((np.array(Patient['P'][0,0][0,0]), np.array(Patient['P'][0,0][0,1])))
        PN = np.column_stack((np.array(Patient['PN'][0,0][0,0]), np.array(Patient['PN'][0,0][0,1])))
        if file == 'd1e396a5-5dfd-4d61-abc8-2142828992cb.mat':
            Pmin = np.zeros(Treal[-1].astype(int)+1)       
        else:
            Pmin = expand(P, Treal)
        umin = expand(u, Treal)
        PNmin = expand(PN, Treal)
        GoalFeed = expand(GoalFeed,Treal)
        ures = resample(u, Treal)
        PNres = resample(PN, Treal)
        Gnt = np.interp(range(Treal[-1,0]+1),Treal[:,0],Greal[:,0])   
    
        """Integrates the Least-Squares Function, Assuming a Linear Glucose Profile"""
        t = range(Lm[0]+1)  
        y_id = np.zeros([Lm[0]+1,6])
        y_id[0,1] = -Pat['pG']*Gnt[0]+(min(Patient['Po'][0,0][0,0],Pat['Pmax'])+Pat['EGP']-Pat['CNS']+PN[0,1])/Pat['Vg']
        y_id[0,2] = (Patient['Uo']+(1-Pat['xl'])*min(max(Pat['uenmin'],Pat['k1'][Dia]*Greal[0]+Pat['k2'][Dia]),Pat['uenmax']))/Pat['Vi']/(Pat['nK']+Pat['nL']+0.5*Pat['nI'])
        y_id[0,3] = y_id[0,2]/2
        y_id[0,0] = Gnt[0]*y_id[0,3]/(1+Pat['alphaG']*y_id[0,3])
        y_id[0,4] = Patient['Po'][0,0][0,0]/Pat['d1']
        y_id[0,5] = Patient['Po'][0,0][0,0]/Pat['d2']
        r = scipy.integrate.ode(ICING2).set_integrator("dopri5")
        for j in range(1,Lm[0]+1):
          r.set_initial_value(y_id[j-1,:],j-1)
          y_id[j,:] = r.integrate(t[j])
        
        for k in range(len(y_id)):
            y_id[k,5] = min(y_id[k,5]*Pat['d2'],Pat['Pmax'])
        Ptot = resample(np.column_stack((np.arange(len(y_id)), y_id[:,5])), Treal)+PNres
        
        """Solves for Hourly SI"""
        tt = Treal[Treal >= 60]
        tt = tt[tt <= tt[-1]-60]
        RawSI = np.zeros([len(tt)])
        for k in range(len(tt)):
          A = y_id[tt[k],0] - y_id[tt[k]-60,0]
          B = -(Gnt[tt[k]]-Gnt[tt[k]-60])+y_id[tt[k],1]-y_id[tt[k]-60,1]
          RawSI[k] = B/A
          
        RawSIplus = np.zeros([len(tt)])
        for k in range(len(tt)):
          A = y_id[tt[k]+60,0] - y_id[tt[k],0]
          B = -(Gnt[tt[k]+60]-Gnt[tt[k]])+y_id[tt[k]+60,1]-y_id[tt[k],1]
          RawSIplus[k] = B/A  

        
        idx = len(Treal)-len(RawSI)  
        GlucData = pd.DataFrame({'Patient': file[:-4], 'Gt': Greal[2+idx:,0], 'Gt-1': Greal[1+idx:-1,0], 'Gt-2': Greal[idx:-2,0], 't': Treal[2+idx:,0], 't-1': Treal[1+idx:-1,0], 't-2': Treal[idx:-2,0], 'ut': ures[1+idx:], 'ut-1': ures[idx:-1], 'ut-2': ures[idx-1:-2], 'Pt': Ptot[idx+1:], 'Pt-1': Ptot[idx:-1], 'Pt-2': Ptot[idx-1:-2], 'SIt': RawSI[2:], 'SIt-1': RawSI[1:-1], 'SIt-2': RawSI[0:-2], 'GF': GoalFeed[tt[2:]], 'Operative': Operative[0,0], 'Gender': Gender[0], 't0': PreProt, 'SIt+1': RawSIplus[2:]}, index = range(len(Greal)-2-idx))    
        Gluc = Gluc.append(GlucData)'''
        GlucData = pd.DataFrame({'Name': file[:-4], 'Admis': Admis, 'Prot': Prot}, index = {1})
        Gluc = Gluc.append(GlucData)
        
'''"""Plots Figures"""
plt.figure()
plt.plot(tt, RawSI)
plt.plot(Treal, Greal*np.mean(RawSI)/np.mean(Greal))
plt.plot(Treal[1:], Pres*np.mean(RawSI)/np.mean(Pres))
plt.plot(Treal[1:], ures*np.mean(RawSI)/np.mean(ures))
plt.plot(Treal, np.zeros(len(Treal)),'k--')
plt.title('Insulin Sensitivity')
plt.xlabel('Time (mins)')
plt.ylabel('Sensitivity')
plt.legend(['SI', 'G', 'P', 'u'])'''

        
Gluc.to_csv('C:\WinPython-64bit-3.5.4.1Qt5\Glucose\GlucData.csv')