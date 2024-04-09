# A transmission network model to calculate inter-regional power flows
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

import numpy as np
from numba import njit

@njit()
def Transmission(solution, output=False, resilience=False):
    """TDC = Network.Transmission(S)"""
    intervals, nodes = (solution.intervals, solution.nodes)
    
    MBaseload = solution.GBaseload # MW
    pkfactor = solution.CPeak / solution.CPeak.sum()
<<<<<<< HEAD
    flexible = solution.flexible.copy()
    MPeak = flexible.reshape(-1,1) * pkfactor.reshape(1,-1) # MW
=======
    MPeak = np.atleast_2d(solution.flexible).T * pkfactor.reshape(1,-1) # MW
>>>>>>> a5b59dbc899c4813e540f8fef2f59f2d51ee0e36

    MLoad_denominator = solution.MLoad.sum(axis=1)
    defactor = np.divide(solution.MLoad, MLoad_denominator.reshape(-1, 1))

    MDeficit = np.atleast_2d(solution.Deficit).T * defactor # MDeficit: EDE(j, t)
    
    if resilience is True:
        MPeakR = np.atleast_2d(solution.RFlexible).T*pkfactor.reshape(1,-1)

    PVl_int, Windl_int = (solution.PVl_int, solution.Windl_int)
    MPV, MWind, MWindR = np.zeros((nodes, intervals)), np.zeros((nodes, intervals)), np.zeros((nodes, intervals))
    
    for i, j in enumerate(solution.Nodel_int):
        MPV[i, :] = solution.GPV[:, np.where(PVl_int==j)[0]].sum(axis=1)
        MWind[i, :] = solution.GWind[:, np.where(Windl_int==j)[0]].sum(axis=1)
        MWindR[i, :] = solution.GWindR[:, np.where(Windl_int==j)[0]].sum(axis=1)
    MPV, MWind, MWindR = MPV.T, MWind.T, MWindR.T # Sij-GPV(t, i), Sij-GWind(t, i), MW

    MPW = MPV + MWind
    MPW_denominator = np.atleast_2d(MPW.sum(axis=1) + 0.00000001).T
    spfactor = np.divide(MPW, MPW_denominator)
    MSpillage = np.atleast_2d(solution.Spillage).T * spfactor # MSpillage: ESP(j, t)

    CPHP = solution.CPHP
    dzsm = CPHP != 0 # divide by zero safe mask
    pcfactor = np.zeros(CPHP.shape)
    pcfactor[dzsm] =  CPHP[dzsm] / CPHP[dzsm].sum(axis=0)
    
    MDischarge = np.atleast_2d(solution.Discharge).T * pcfactor# MDischarge: DPH(j, t)
    MCharge = np.atleast_2d(solution.Charge).T * pcfactor # MCharge: CHPH(j, t)
    MP2V = np.atleast_2d(solution.P2V).T * pcfactor
    
    CDP = solution.CDP
    dzsm = CDP!=0
    pcfactorD = np.zeros(CDP.shape)
    pcfactorD[dzsm] = pcfactorD[dzsm] / CDP[dzsm].sum()
    MChargeD = np.atleast_2d(solution.ChargeD).T * pcfactorD # MCharge: CHPH(j, t)

    MImport = solution.MLoad + MCharge + MChargeD + MSpillage - MPV - MWind - MBaseload - MPeak - MDischarge + MP2V - MDeficit # EIM(t, j), MW
              
    FQ = -1 * MImport[:, np.where(solution.Nodel_int==0)[0][0]] if 0 in solution.Nodel_int else np.zeros(intervals, dtype=np.float64)
    AS = -1 * MImport[:, np.where(solution.Nodel_int==2)[0][0]] if 2 in solution.Nodel_int else np.zeros(intervals, dtype=np.float64)
    SW = MImport[:, np.where(solution.Nodel_int==7)[0][0]] if 7 in solution.Nodel_int else np.zeros(intervals, dtype=np.float64)
    TV = -1 * MImport[:, np.where(solution.Nodel_int==5)[0][0]]

    NQ = MImport[:, np.where(solution.Nodel_int==3)[0][0]] - FQ
    NV = MImport[:, np.where(solution.Nodel_int==6)[0][0]] - TV

    NS = -1. * MImport[:, np.where(solution.Nodel_int==1)[0][0]] - NQ - NV
    NS1 = MImport[:, np.where(solution.Nodel_int==4)[0][0]] - AS + SW
    # assert abs(NS - NS1).max()<=0.1, print(abs(NS - NS1).max())
    
    TDC = np.stack((FQ, NQ, NS, NV, AS, SW, TV), axis=1) # TDC(t, k), MW
    
    if output:
        MStorage = np.atleast_2d(solution.Storage).T * pcfactor # SPH(t, j), MWh
        MDischargeD = np.atleast_2d(solution.DischargeD).T * pcfactorD  # MDischarge: DD(j, t)
        MStorageD = np.atleast_2d(solution.StorageD).T * pcfactorD  # SD(t, j), MWhD
    
<<<<<<< HEAD
        solution.MPV, solution.MWind, solution.MBaseload, solution.MPeak = (MPV, MWind, MBaseload, MPeak)
=======
        solution.MPV, solution.MWind, solution.MWindR, solution.MBaseload, solution.MPeak = (MPV, MWind, MWindR, MBaseload, MPeak)
>>>>>>> a5b59dbc899c4813e540f8fef2f59f2d51ee0e36
        solution.MDischarge, solution.MCharge, solution.MStorage, solution.MP2V = (MDischarge, MCharge, MStorage, MP2V)
        solution.MDischargeD, solution.MChargeD, solution.MStorageD = (MDischargeD, MChargeD, MStorageD)
        solution.MDeficit, solution.MSpillage = (MDeficit, MSpillage)
        
        if resilience is True:
            solution.MPeakR = MPeakR
    
    return TDC

    
    
