# To simulate energy supply-demand balance based on long-term, high-resolution chronological data
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

import numpy as np
from numba import njit

@njit
def Reliability(solution, flexible):
    """Deficit = Simulation.Reliability(S, hydro=...)"""

    Netload = (solution.MOLoad.sum(axis=1) 
               - solution.GPV.sum(axis=1) 
               - solution.GOnsW.sum(axis=1) 
               - solution.GOffW.sum(axis=1) 
               - solution.GBaseload.sum(axis=1)
               ) - flexible

    Pcapacity = solution.CPHP.sum() * 1000 # S-CPHP(j), GW to MW
    Scapacity = solution.CPHS * 1000 # 

    solution.Discharge = np.zeros(solution.intervals)
    solution.Charge = np.zeros(solution.intervals)
    solution.Storage = np.zeros(solution.intervals)
    solution.Storage[-1] = 0.5*Scapacity
    for t in range(solution.intervals):
        solution.Discharge[t] = np.minimum(np.minimum(np.maximum(0, Netload[t]), Pcapacity), solution.Storage[t-1] / solution.resolution)
        solution.Charge[t] = np.minimum(np.minimum(-1 * np.minimum(0, Netload[t]), Pcapacity), (Scapacity - solution.Storage[t-1]) / solution.efficiency / solution.resolution)
        solution.Storage[t] = solution.Storage[t-1] - solution.Discharge[t] * solution.resolution + solution.Charge[t] * solution.resolution * solution.efficiency

    solution.Deficit = np.maximum(Netload - solution.Discharge, 0)
    solution.Spillage = - np.minimum(Netload + solution.Charge, 0)
    solution.flexible = flexible
    
    return solution.Deficit
