clear
clc
close all
load('C:\WinPython-64bit-3.5.4.1Qt5\Glucose\SP5012.mat')

CNS = PatientStruct.CNS
Diabetic = PatientStruct.Diabetic
EGP = PatientStruct.EGP
GoalFeed = PatientStruct.GoalFeed
Greal = PatientStruct.Greal
P = PatientStruct.P
PN = PatientStruct.PN
Pmax = PatientStruct.Pmax
Po = PatientStruct.Po
Treal = PatientStruct.Treal
Ueninit = PatientStruct.Ueninit
Uo = PatientStruct.Uo
Vg = PatientStruct.Vg
Vi = PatientStruct.Vi
alpha_G = PatientStruct.alpha_G
alpha_I = PatientStruct.alpha_I
d1 = PatientStruct.d1
d2 = PatientStruct.d2
gamma = PatientStruct.gamma
k1 = PatientStruct.k1
k2 = PatientStruct.k2
maxIter = PatientStruct.maxIter
nC = PatientStruct.nC
nI = PatientStruct.nI
nK = PatientStruct.nK
nL = PatientStruct.nL
pG = PatientStruct.pG
rawSI = PatientStruct.rawSI
u = PatientStruct.u
uenmax = PatientStruct.uenmax
uenmin = PatientStruct.uenmin
varEGP = PatientStruct.varEGP
xl = PatientStruct.xl
StartDelayHrs = PatientStruct.StartDelayHrs
T = TimeSoln.T
GIQ = TimeSoln.GIQ
Pt = TimeSoln.P

EGP = cell2mat(EGP)
P = cell2mat(P)
u = cell2mat(u)
rawSI = cell2mat(rawSI)
PN = cell2mat(PN)

clear TimeSoln PatientStruct