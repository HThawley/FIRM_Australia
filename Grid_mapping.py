# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 09:22:08 2024

@author: u6942852
"""

import numpy as np 
from Setup import *
from math import log
import csv

from multiprocessing import Pool, cpu_count


# x0 = np.genfromtxt('costOptimisationResults/Optimisation_resultx{}-None.csv'.format(scenario), delimiter=',', dtype=float)
x0 = np.zeros(len(bounds))

inc = 10
dp = round(log(inc, 10))

x0 = np.round(x0, dp)

indexlength = len(x0)
ub[-1] = 1000

def variable_stepping(x, i):
    print(i)
    xi = x.copy()
    xi[i] = 0
    
    mask = np.zeros(x.shape)
    mask[i] = 1
    
    for j in range(0, int(ub[i]+inc), inc):
        xk = xi + j*mask
        
        with open('Results/mapping{}.csv'.format(scenario), 'a', newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([F(xk)] + list(xk))
            
        if i+1 < indexlength:
            variable_stepping(xi, i+1)

    # return res

with open('Results/mapping{}.csv'.format(scenario), 'w', newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Objective'] + ['dvar']*indexlength)
    
    
variable_stepping(x0,0)
        
# with Pool(processes=min(indexlength, cpu_count())) as processPool:
#     result = processPool.starmap(variable_stepping, [(x0, i) for i in range(indexlength)])
    
#     result = np.concatenate(result)



