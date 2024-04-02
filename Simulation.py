# To simulate energy supply-demand balance based on long-term, high-resolution chronological data
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

import numpy as np
from numba import njit

@njit()
def Reliability(solution, flexible, start=None, end=None, output=False):
    """Deficit = Simulation.Reliability(S, hydro=...)"""
    intervals = solution.intervals

    Netload = (solution.MLoad.sum(axis=1) - solution.GPV.sum(axis=1) - solution.GWind.sum(axis=1) - solution.GBaseload.sum(axis=1))[start:end] - flexible # Sj-ENLoad(j, t), MW

    if output: 
        solution.flexible = flexible # MW

    Pcapacity = solution.CPHP.sum() * 1000 # S-CPHP(j), GW to MW
    Scapacity = solution.CPHS * 1000 # S-CPHS(j), GWh to MWh
    PcapacityD = solution.CDP.sum() * 1000 # S-CDP(j), GW to MW
    ScapacityD = solution.CDS.sum() * 1000 # S-CDS(j), GWh to MWh
    efficiency, efficiencyD, resolution = (solution.efficiency, solution.efficiencyD, solution.resolution)

    Discharge, Charge, Storage, DischargeD, ChargeD, StorageD, P2V = [np.zeros(intervals)]*7

    ConsumeD = solution.MLoadD.sum(axis=1)[start:end] * efficiencyD

    for t in range(intervals):

        Netloadt = Netload[t]
        
        Storaget_1 = Storage[t-1] if t>0 else 0.5 * Scapacity
        StorageDt_1 = StorageD[t-1] if t>0 else 0.5 * ScapacityD

        Discharget = np.minimum(np.minimum(np.maximum(0, Netloadt), Pcapacity), Storaget_1 / resolution)
        Charget = np.minimum(np.minimum(-1 * np.minimum(0, Netloadt), Pcapacity), (Scapacity - Storaget_1) / efficiency / resolution)
        Storaget = Storaget_1 - Discharget * resolution + Charget * resolution * efficiency
        
        ConsumeDt = ConsumeD[t]

        DischargeDt = np.minimum(ConsumeDt, StorageDt_1 / resolution)
        ChargeDt = np.minimum(np.minimum(-1 * np.minimum(0, Netloadt + Charget), PcapacityD), (ScapacityD - StorageDt_1) / efficiencyD / resolution)
        StorageDt = StorageDt_1 - DischargeDt * resolution + ChargeDt * resolution * efficiencyD

        diff = ConsumeDt - DischargeDt
        P2Vt = np.minimum(diff / efficiencyD, Pcapacity - Discharget - Charget) if diff > 0 and Storaget / resolution > diff / efficiencyD else 0

        Discharge[t] = Discharget + P2Vt
        P2V[t] = P2Vt
        Charge[t] = Charget
        Storage[t] = Storaget - P2Vt * resolution        

        DischargeD[t] = DischargeDt
        ChargeD[t] = ChargeDt
        StorageD[t] = StorageDt
        
    Deficit = np.maximum(Netload - Discharge + P2V, 0)
    DeficitD = ConsumeD - DischargeD - P2V * efficiencyD
    Spillage = -1 * np.minimum(Netload + Charge + ChargeD, 0)


    # assert 0 <= int(np.amax(Storage)) <= Scapacity, 'Storage below zero or exceeds max storage capacity'
    # assert 0 <= int(np.amax(StorageD)) <= ScapacityD, 'StorageD below zero or exceeds max storage capacity'
    # assert np.amin(Deficit) >= 0, 'Deficit below zero: {}'.format(np.amin(Deficit))
    # assert np.amin(DeficitD) > -0.1, 'DeficitD below zero: {}'.format(np.amin(DeficitD))
    # assert np.amin(Spillage) >= 0, 'Spillage below zero: {}'.format(np.amin(Spillage))
    
    if output:
        solution.Discharge, solution.Charge, solution.Storage, solution.P2V = (Discharge, Charge, Storage, P2V)
        solution.DischargeD, solution.ChargeD, solution.StorageD = (DischargeD, ChargeD, StorageD)
        solution.Deficit, solution.DeficitD, solution.Spillage = (Deficit, DeficitD, Spillage)
    
    return Deficit, DeficitD

