# To simulate energy supply-demand balance based on long-term, high-resolution chronological data
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

import numpy as np
from Simulation import Reliability
from numba import njit, jit

@jit()
def Resilience(solution, flexible, start=None, end=None, output=True):
    """ """
    windDiff, eventDur, eventZoneIndx = solution.WindDiff[start:end], solution.eventDur, solution.eventZoneIndx
    
    solution.flexible = flexible

    Netload = (solution.MLoad.sum(axis=1) - solution.GPV.sum(axis=1) - solution.GWind.sum(axis=1) - solution.GBaseload.sum(axis=1))[start:end] - flexible # Sj-ENLoad(j, t), MW
    length = len(Netload)
    
    Deficit, DeficitD = Reliability(solution, flexible, start, end, output=True)
    
    if eventZoneIndx[0] >= 0: 
        if len(eventZoneIndx) > 1 and solution.logic == 'and':
            eventZoneIndx = np.array([-1])
            windDiff = windDiff.sum(axis=1).reshape((-1,1))
        
        if len(eventZoneIndx) == 1: 
            RNetload = Netload - windDiff.sum(axis=1)
    
            # State of charge is taken as the state of charge {eventDuration} steps ago 
            # +/- the charging that occurs under the modified generation capacity   
            storageAdj = np.zeros(length)
            for i in eventZoneIndx:
                Rgen = np.hstack((np.zeros(eventDur - 1), np.clip(windDiff[:,i]-solution.Spillage, None, 0)))
                adj = np.lib.stride_tricks.sliding_window_view(Rgen, eventDur).sum(axis=1)
                storageAdj = storageAdj + adj
            
            RDeficit, RDeficitD = Resilience_wrapped(storageAdj, RNetload, solution, start, end, None, output)
            RDeficit, RDeficitD = RDeficit.reshape(-1,1), RDeficitD.reshape(-1, 1)
        
        if len(eventZoneIndx) > 1:
            assert solution.logic == 'xor'
            
            zs = (length, len(eventZoneIndx))
            RDeficit, RDeficitD = np.zeros(zs), np.zeros(zs)
            
            if output is True:
                solution.RDischarge = np.zeros(zs)
                solution.RCharge = np.zeros(zs)
                solution.RStorage = np.zeros(zs)
                solution.RP2V = np.zeros(zs)
                solution.RDischargeD = np.zeros(zs)
                solution.RChargeD = np.zeros(zs)
                solution.RStorageD = np.zeros(zs)
                solution.RDeficit = np.zeros(zs)
                solution.RDeficitD = np.zeros(zs)
                solution.RSpillage = np.zeros(zs)

            for i, z in enumerate(eventZoneIndx):
                Rgen = np.hstack((np.zeros(eventDur - 1), 
                                  np.clip(windDiff[:,z]-solution.Spillage, None, 0)))
                storageAdj = np.lib.stride_tricks.sliding_window_view(Rgen, eventDur).sum(axis=1)
                
                RDeficit[:, i], RDeficitD[:, i] = Resilience_wrapped(
                    storageAdj, Netload - windDiff[:,z], solution, start, end, i, output)
                
    else: 
        if output is True:
            solution.RDischarge = np.atleast_2d(solution.Discharge).T
            solution.RCharge = np.atleast_2d(solution.Charge).T
            solution.RStorage = np.atleast_2d(solution.Storage).T
            solution.RP2V = np.atleast_2d(solution.P2V).T
            solution.RDischargeD = np.atleast_2d(solution.DischargeD).T
            solution.RChargeD = np.atleast_2d(solution.ChargeD).T
            solution.RStorageD = np.atleast_2d(solution.StorageD).T
            solution.RDeficit = np.atleast_2d(solution.Deficit).T
            solution.RDeficitD = np.atleast_2d(solution.DeficitD).T
            solution.RSpillage = np.atleast_2d(solution.Spillage).T
    
    return Deficit, DeficitD, RDeficit, RDeficitD
        
@njit()
def Resilience_wrapped(storageAdj, RNetload, solution, start, end, j, output):
    """Re-simulate with resilience losses due to windevent, and including storage depletion"""
    length = len(RNetload)
    
    Pcapacity = solution.CPHP.sum() * 1000 # S-CPHP(j), GW to MW
    Scapacity = solution.CPHS * 1000 # S-CPHS(j), GWh to MWh
    PcapacityD = solution.CDP.sum() * 1000 # S-CDP(j), GW to MW
    ScapacityD = solution.CDS.sum() * 1000 # S-CDS(j), GWh to MWh
    efficiency, efficiencyD, resolution = (solution.efficiency, solution.efficiencyD, solution.resolution)
    
    ConsumeD = solution.MLoadD.sum(axis=1)[start:end] * efficiencyD
    
    RDischarge = np.zeros(length)
    RCharge = np.zeros(length)
    RStorage = np.zeros(length)
    RP2V = np.zeros(length)
    RDischargeD = np.zeros(length)
    RChargeD = np.zeros(length)
    RStorageD = np.zeros(length)
    RDeficit = np.zeros(length)
    RDeficitD = np.zeros(length)
    RSpillage = np.zeros(length)
    
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

    if output is True:
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
    return RDeficit, RDeficitD
