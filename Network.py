# A transmission network model to calculate inter-regional power flows
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

import numpy as np
from numba import njit

@njit()
def Transmission(solution, output=False):
    """TDC = Network.Transmission(S)"""
    intervals, nodes = (solution.intervals, solution.nodes)
    
    MBaseload = solution.GBaseload # MW
    pkfactor = solution.CPeak / solution.CPeak.sum()
    flexible = solution.flexible.copy()
    MPeak = flexible.reshape(-1,1) * pkfactor.reshape(1,-1) # MW
    
    try: 
        if solution.RFlexible.size != 0:
            RMPeak = np.atleast_2d(solution.RFlexible).T*pkfactor.reshape(1,-1)
    except Exception: # NameError but jit can't type excpetions
        pass

    MLoad_denominator = solution.MLoad.sum(axis=1)
    defactor = np.divide(solution.MLoad, MLoad_denominator.reshape(-1, 1))

    MDeficit = solution.Deficit.copy() # avoids numba error with reshaping below (also MSpillage, MCharge, MDischarge)
    MDeficit = MDeficit.reshape(-1, 1) * defactor # MDeficit: EDE(j, t)

    PVl_int, Windl_int = (solution.PVl_int, solution.Windl_int)
    MPV, MWind = [np.zeros((nodes, intervals))] * 2
    for i, j in enumerate(solution.Nodel_int):
        MPV[i, :] = solution.GPV[:, np.where(PVl_int==j)[0]].sum(axis=1)
        MWind[i, :] = solution.GWind[:, np.where(Windl_int==j)[0]].sum(axis=1)
    MPV, MWind = MPV.T, MWind.T # Sij-GPV(t, i), Sij-GWind(t, i), MW

    MPW = MPV + MWind
    MPW_denominator = np.atleast_2d(MPW.sum(axis=1) + 0.00000001).T
    spfactor = np.divide(MPW, MPW_denominator)
    MSpillage = solution.Spillage.copy()
    MSpillage = MSpillage.reshape(-1, 1) * spfactor # MSpillage: ESP(j, t)

    CPHP = solution.CPHP
    dzsm = CPHP != 0 # divide by zero safe mask
    pcfactor = np.zeros(CPHP.shape)
    pcfactor[dzsm] =  CPHP[dzsm] / CPHP[dzsm].sum(axis=0)
    
    MCharge, MDischarge, MP2V = solution.Charge.copy(), solution.Discharge.copy(), solution.P2V.copy()
    MDischarge = (MDischarge.reshape(-1, 1) * pcfactor)# MDischarge: DPH(j, t)
    MCharge = (MCharge.reshape(-1, 1) * pcfactor) # MCharge: CHPH(j, t)
    MP2V = (MP2V.reshape(-1, 1) * pcfactor)
    
    CDP, MChargeD = solution.CDP, solution.ChargeD.copy()
    dzsm = CDP!=0
    pcfactorD = np.zeros(CDP.shape)
    pcfactorD[dzsm] = pcfactorD[dzsm] / CDP[dzsm].sum()
    MChargeD = (MChargeD.reshape(-1, 1) * pcfactorD) # MCharge: CHPH(j, t)

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
        Storage, DischargeD, StorageD = solution.Storage.copy(), solution.DischargeD.copy(), solution.StorageD.copy()
        MStorage = Storage.reshape(-1,1) * pcfactor # SPH(t, j), MWh
        MDischargeD = DischargeD.reshape(-1,1) * pcfactorD  # MDischarge: DD(j, t)
        MStorageD = StorageD.reshape(-1,1) * pcfactorD  # SD(t, j), MWhD
    
        solution.MPV, solution.MWind, solution.MBaseload, solution.MPeak, solution.RMPeak = (MPV, MWind, MBaseload, MPeak, RMPeak)
        solution.MDischarge, solution.MCharge, solution.MStorage, solution.MP2V = (MDischarge, MCharge, MStorage, MP2V)
        solution.MDischargeD, solution.MChargeD, solution.MStorageD = (MDischargeD, MChargeD, MStorageD)
        solution.MDeficit, solution.MSpillage = (MDeficit, MSpillage)
    return TDC

# if __name__ == '__main__':
#     from Input import suffix, scenario, CPeak, intervals, Solution
#     from DeficitSimulation import DeficitSimulation
#     from CoSimulation import Resilience
    
#     costCapacities = np.genfromtxt('CostOptimisationResults/Optimisation_resultx{}-None.csv'.format(scenario), delimiter=',')
#     capacities = np.genfromtxt('Results/Optimisation_resultx'+suffix, delimiter=',')
#     flexible = CPeak.sum() * pow(10, 3) * np.ones(intervals)
    
#     solution = Solution(capacities)
#     Deficit, DeficitD, RDeficit, RDeficitD = Resilience(solution, flexible=flexible)
    
#     topDeficitIndx = np.flip(np.argpartition(RDeficit, -1)[-1:])
    
#     for n, indx in enumerate(topDeficitIndx):
#         solution = Solution(capacities)
#         solution = DeficitSimulation(solution, flexible, RSim=indx, output='solution')
    
#     solution.TDC, solution.TDCR = Transmission(solution, True, True)
    
    
