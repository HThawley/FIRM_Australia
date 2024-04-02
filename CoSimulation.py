# To simulate energy supply-demand balance based on long-term, high-resolution chronological data
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

import numpy as np
from Simulation import Reliability
from numba import njit, jit

@jit()
def Resilience(solution, flexible, start=None, end=None):
    """ """
    intervals = solution.intervals
    windDiff, eventDur, eventZoneIndx = solution.WindDiff[start:end], solution.eventDur, solution.eventZoneIndx
    
    solution.flexible = flexible

    Netload = (solution.MLoad.sum(axis=1) - solution.GPV.sum(axis=1) - solution.GWind.sum(axis=1) - solution.GBaseload.sum(axis=1))[start:end] - flexible # Sj-ENLoad(j, t), MW
    
    Reliability(solution, flexible, start, end, output=True)
    
    if eventZoneIndx[0] > 0: 
        if len(eventZoneIndx) > 1 and solution.logic == 'and':
            eventZoneIndx = np.array([-1])
            windDiff = windDiff.sum(axis=1).reshape((-1,1))
        
        if len(eventZoneIndx) == 1: 
            RNetload = Netload - windDiff.sum(axis=1)
    
            # State of charge is taken as the state of charge {eventDuration} steps ago 
            # +/- the charging that occurs under the modified generation capacity   
            storageAdj = np.zeros(intervals)
            for i in eventZoneIndx:
                Rgen = np.hstack((np.zeros(eventDur - 1), np.clip(windDiff[:,i]-solution.Spillage, None, 0)))
                adj = np.lib.stride_tricks.sliding_window_view(Rgen, eventDur).sum(axis=1)
                storageAdj = storageAdj + adj
            
            Resilience_wrapped(storageAdj, RNetload, solution, start, end)
            
        
        if len(eventZoneIndx) > 1:
            assert solution.logic == 'xor'
            
            for i in eventZoneIndx:
                # z =  # function only recognises full length windows -> pad with zeros
                # c =  # excess generation
                Rgen = np.hstack((np.zeros(eventDur - 1), np.clip(windDiff[:,i]-solution.Spillage, None, 0)))
                storageAdj = np.lib.stride_tricks.sliding_window_view(Rgen, eventDur).sum(axis=1)
                
                RNetload = Netload - windDiff[:,i]
            
                Resilience_wrapped(storageAdj, RNetload, solution, start, end, i)
                
    # else: 
    #     solution.RDischarge, solution.RCharge, solution.RStorage, solution.RP2V = (
    #         solution.Discharge.reshape(-1,1), solution.Charge.reshape(-1,1), solution.Storage.reshape(-1,1), solution.P2V.reshape(-1,1))
    #     solution.RDischargeD, solution.RChargeD, solution.RStorageD = (
    #         solution.DischargeD.reshape(-1,1), solution.ChargeD.reshape(-1,1), solution.StorageD.reshape(-1,1))
    #     solution.RDeficit, solution.RDeficitD, solution.RSpillage = (
    #         solution.Deficit.reshape(-1,1), solution.DeficitD.reshape(-1,1), solution.Spillage.reshape(-1,1)) 
        
    return solution.Deficit, solution.DeficitD, solution.RDeficit, solution.RDeficitD
        
@njit()
def Resilience_wrapped(storageAdj, RNetload, solution, start=None, end=None, j=None):
    """Re-simulate with resilience losses due to windevent, and including storage depletion"""
    intervals = solution.intervals
    
    Pcapacity = solution.CPHP.sum() * 1000 # S-CPHP(j), GW to MW
    Scapacity = solution.CPHS * 1000 # S-CPHS(j), GWh to MWh
    PcapacityD = solution.CDP.sum() * 1000 # S-CDP(j), GW to MW
    ScapacityD = solution.CDS.sum() * 1000 # S-CDS(j), GWh to MWh
    efficiency, efficiencyD, resolution = (solution.efficiency, solution.efficiencyD, solution.resolution)

    
    ConsumeD = solution.MLoadD.sum(axis=1)[start:end] * efficiencyD
    
    (RDischarge, RCharge, RStorage, RP2V,  RDischargeD, RChargeD, RStorageD, 
     RDeficit, RDeficitD, RSpillage) = [np.zeros((intervals))] * 10
    
    #Resimulate
    Storage_1, StorageD_1 = np.roll(solution.Storage, 1), np.roll(solution.StorageD, 1)
    Storage_1[0], StorageD_1[0] = 0.5 * Scapacity, 0.5 * ScapacityD
    
    RStorage_1 = np.maximum(0, Storage_1 + storageAdj)
    RStorageD_1 = np.maximum(0, StorageD_1 + np.clip(Storage_1 + storageAdj, None, 0))
    RDeficit = np.maximum(0, -(StorageD_1 + Storage_1 + storageAdj))
    
    RDischarge = np.minimum(np.minimum(np.maximum(0, RNetload), Pcapacity), RStorage_1 / resolution)
    RCharge = np.minimum(np.minimum(-1 * np.minimum(0, RNetload), Pcapacity), (Scapacity - RStorage_1) /efficiency /resolution)        
    RStorage = RStorage_1 - RDischarge * resolution + RCharge * resolution * efficiency
    
    RDischargeD = np.minimum(ConsumeD, StorageD_1 / resolution)
    RChargeD = np.minimum(np.minimum(-1 * np.minimum(0, RNetload + RCharge), PcapacityD), (ScapacityD - RStorageD_1) /efficiencyD /resolution)
    RStorageD = RStorageD_1 - RDischargeD * resolution + RChargeD * resolution * efficiencyD

    Rdiff = ConsumeD - RDischargeD 
    RP2V = np.minimum(Rdiff / efficiencyD, Pcapacity - RDischarge - RCharge) 
    RP2V[np.where((Rdiff <= 0) | (RStorage / resolution <= Rdiff / efficiencyD))] = 0

    RDischarge = RDischarge + RP2V
    RStorage = RStorage - RP2V * resolution 
    
    RDeficit = np.maximum(RNetload - RDischarge + RP2V, 0)
    RDeficitD = ConsumeD - RDischargeD - RP2V * efficiencyD
    RSpillage = -1 * np.minimum(RNetload + RCharge + RChargeD, 0)

    
    if j is not None: 
        solution.RDischarge[:,j], solution.RCharge[:,j], solution.RStorage[:,j], solution.RP2V[:,j] = (RDischarge, RCharge, RStorage, RP2V)
        solution.RDischargeD[:,j], solution.RChargeD[:,j], solution.RStorageD[:,j] = (RDischargeD, RChargeD, RStorageD)
        solution.RDeficit[:,j], solution.RDeficitD[:,j], solution.RSpillage[:,j] = (RDeficit, RDeficitD, RSpillage)                 
    else: 
        solution.RDischarge, solution.RCharge, solution.RStorage, solution.RP2V = (
            RDischarge.reshape(-1,1), RCharge.reshape(-1,1), RStorage.reshape(-1,1), RP2V.reshape(-1,1))
        solution.RDischargeD, solution.RChargeD, solution.RStorageD = (
            RDischargeD.reshape(-1,1), RChargeD.reshape(-1,1), RStorageD.reshape(-1,1))
        solution.RDeficit, solution.RDeficitD, solution.RSpillage = (
            RDeficit.reshape(-1,1), RDeficitD.reshape(-1,1), RSpillage.reshape(-1,1)) 
