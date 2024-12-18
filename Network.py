# A transmission network model to calculate inter-regional power flows
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

import numpy as np
from numba import njit

@njit
def Transmission(solution):
    """TDC = Network.Transmission(S)"""

    solution.MPV   = np.empty((solution.intervals, solution.nodes), dtype=np.float64)
    solution.MOnsW = np.empty((solution.intervals, solution.nodes), dtype=np.float64)
    solution.MOffW = np.empty((solution.intervals, solution.nodes), dtype=np.float64)
    
    for i, j in enumerate(solution.Nodel_int):
        solution.MPV[:,   i] = solution.GPV[:,   np.where(solution.PVl_int  ==j)[0]].sum(axis=1)
        solution.MOnsW[:, i] = solution.GOnsW[:, np.where(solution.OnsWl_int==j)[0]].sum(axis=1)
        solution.MOffW[:, i] = solution.GOffW[:, np.where(solution.OffWl_int==j)[0]].sum(axis=1)
    MPW = solution.MPV + solution.MOnsW + solution.MOffW

    solution.MSpillage = np.atleast_2d(solution.Spillage / MPW.sum(axis=1)).T * MPW
    solution.MPeak = np.atleast_2d(solution.flexible).T * solution.CPeak / solution.CPeak.sum()
    solution.MDeficit = np.atleast_2d(solution.Deficit / solution.MLoad.sum(axis=1)).T * solution.MLoad 

    pcfactor =  np.atleast_2d(solution.CPHP / solution.CPHP.sum(axis=0)).T
    solution.MDischarge = (solution.Discharge * pcfactor).T
    solution.MCharge = (solution.Charge * pcfactor).T
    solution.MStorage = (solution.Storage * pcfactor).T

    solution.MImport = (solution.MLoad + solution.MCharge + solution.MSpillage \
              - MPW - solution.GBaseload - solution.MPeak - solution.MDischarge - solution.MDeficit).T

    solution.TDC = np.zeros((7, solution.intervals), np.float64)
    if 0 in solution.Nodel_int: solution.TDC[0] = - solution.MImport[np.where(solution.Nodel_int==0)[0][0]]  
    if 2 in solution.Nodel_int: solution.TDC[4] = - solution.MImport[np.where(solution.Nodel_int==2)[0][0]]  
    if 7 in solution.Nodel_int: solution.TDC[5] =   solution.MImport[np.where(solution.Nodel_int==7)[0][0]] 
    solution.TDC[6] = - solution.MImport[np.where(solution.Nodel_int==5)[0][0]]
    solution.TDC[1] =   solution.MImport[np.where(solution.Nodel_int==3)[0][0]] - solution.TDC[0]
    solution.TDC[3] =   solution.MImport[np.where(solution.Nodel_int==6)[0][0]] - solution.TDC[6]
    solution.TDC[2] = - solution.MImport[np.where(solution.Nodel_int==1)[0][0]] - solution.TDC[1] - solution.TDC[3]
    solution.TDC = solution.TDC.T
    return solution.TDC

