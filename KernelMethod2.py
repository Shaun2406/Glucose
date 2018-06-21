# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 12:06:41 2018

@author: smd118
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats
from time import time
import multiprocessing as mp

plt.close("all")

NUM_THREADS = 8
'''
class Threaded_Solver(threading.Thread):
    done = False
    progress = 0
    name = ""
    density_func = []
    
    def __init__(self, name, x_min, x_max, grid_pts, means, trf, measured_trf, sigma):
        threading.Thread.__init__(self)
        
        self.name = name
        self.x_min = x_min
        self.x_range = x_max - x_min
        self.grid_pts = grid_pts
        self.means = means
        self.trf = trf
        self.measured_trf = measured_trf
        self.sigma = sigma
        
        self.density_func = np.zeros([len(grid_pts[0]), len(grid_pts[1])])
    
    def run(self):
        print("yay")
        for i in range(self.x_min, self.x_min + self.x_range):
            if (i%50 == 0):
                self.progress = (i-self.x_min)/self.x_range*100
            self.bivar_norm(i)
        self.progress = 100
        self.done = True
'''
def bivar_norm(measured_pt, sigma, means, grid_pts, trf):
    density_func = np.zeros([len(grid_pts[0]), len(grid_pts[1])])
    for x in range(len(grid_pts[0])):
        for y in range(len(grid_pts[1])):
            xy = np.matmul([grid_pts[0][x]-means[0], grid_pts[1][y]-means[1]], trf)
            density_func[y,x] = 1/(2*np.pi*sigma**2)*np.exp(-1/2*((xy[0]-measured_pt[0])**2/sigma**2+(xy[1]-measured_pt[1])**2/sigma**2))

def load_data():
    '''LOADING DATA'''
    GlucData = pd.read_csv('GlucDataOverall.csv')
    GlucData = GlucData.drop(['Unnamed: 0', 'Unnamed: 0.1', 'Operative', 'Patient', 't0', 'GF'], axis = 1)
    GlucData['Gender'] = GlucData['Gender'] == 'female'
    GlucData['Gender'] = GlucData['Gender'].astype(int)
    
    '''LOGGING RELEVANT DATA'''
    GlucData['Gt'] = np.log10(GlucData['Gt'])
    GlucData = GlucData[np.isnan(GlucData['Gt']) == 0]
    
    GlucData['SIt+1'] = np.log10(GlucData['SIt+1'])
    GlucData = GlucData[np.isnan(GlucData['SIt+1']) == 0]
    
    GlucData['SIt'] = np.log10(GlucData['SIt'])
    GlucData = GlucData[np.isnan(GlucData['SIt']) == 0]
    
    GlucData = GlucData.reset_index()
    return GlucData

def transform(GlucData):
    '''Create an Ortho-Normalised Matrix Xdec - 2D, for w(x)'''
    Xin = GlucData.loc[:,['SIt', 'Gt']].values
    Cin = np.cov(np.transpose(Xin))
    Rin = np.linalg.cholesky(Cin)
    Ain = np.linalg.inv(np.transpose(Rin))
    detAin = np.linalg.det(Ain)
    Xin0 = Xin - np.mean(Xin, 0)
    Xindec = np.matmul(Xin0, Ain)
    
    return (Xin, Xindec, detAin, Ain)

if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    GlucData = load_data()
    measured, measured_trf, trf_det, trf = transform(GlucData)
    
    sigma = pd.read_csv('KernelSigma.csv')
    sigma = sigma.drop(['Unnamed: 0'], axis = 1)
    sigma = np.array(sigma)
    
    res = 10
    
    grid_pts = [np.linspace(np.min(measured[:,0]), np.max(measured[:,0]), res),
                np.linspace(np.min(measured[:,1]), np.max(measured[:,1]), res)]
    density_func = np.zeros([res]*2)
    means = np.mean(measured, 0)
    
    
    #bivar_norm(measured_pt, sigma, means, grid_pts, trf)
    
'''
num_ops = len(measured)
ops_per_thread = num_ops/NUM_THREADS
print(ops_per_thread)

solvers = [Threaded_Solver(i, round(ops_per_thread*i), round(ops_per_thread*(i+1)), grid_pts, means, trf, measured_trf, sigma) for i in range(NUM_THREADS)]
print("starting the thing")
for solver in solvers:
    solver.start()
print("finishing the thing")

start = time()
while not all(solver.done for solver in solvers):
    for solver in solvers:
        print(solver.name, solver.progress)
    print()
    sleep(0.5)
    
out = sum(solver.density_func for solver in solvers)
end = time()
print(end-start)'''
'''
for i in range(len(Xin)):
    PDF = PDF + bivar_norm(SItx, Gtx, i)
    if i % 1000 == 0:
        print(i)'''
'''np.savetxt('C:\WinPython-64bit-3.5.4.1Qt5\Glucose\WX.txt', PDF, delimiter=',')'''
