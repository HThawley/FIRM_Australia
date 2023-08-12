# To simulate energy supply-demand balance based on long-term, high-resolution chronological data
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

import numpy as np
from Simulation import Reliability

def Resilience(solution, flexible, start=None, end=None):
    """Deficit = Simulation.Resilience(S, hydro=...)"""

    windDiff, stormDur, stormZone = solution.WindDiff, solution.stormDur, solution.stormZone
    
    solution.flexible = flexible # MW

    Pcapacity = sum(solution.CPHP) * pow(10, 3) # S-CPHP(j), GW to MW
    Scapacity = solution.CPHS * pow(10, 3) # S-CPHS(j), GWh to MWh
    PcapacityD = sum(solution.CDP) * pow(10, 3) # S-CDP(j), GW to MW
    ScapacityD = sum(solution.CDS) * pow(10, 3) # S-CDS(j), GWh to MWh
    efficiency, efficiencyD, resolution = (solution.efficiency, solution.efficiencyD, solution.resolution)

    ConsumeD = solution.MLoadD.sum(axis=1)[start:end] * efficiencyD

    (Netload, Storage, StorageD, Deficit, DeficitD, Spillage, Discharge, DischargeD, 
     Charge, ChargeD, P2V) = Reliability(solution, flexible, start, end, output=False, resilience = True)

    try:
        RNetload = Netload + windDiff.sum(axis=1) 
    except np.AxisError: 
        RNetload = Netload + windDiff

# =============================================================================
#         # Re-simulate with resilience losses due to windstorm, and including storage depletion
# =============================================================================
    if stormZone is not None:    
        # State of charge is taken as the state of charge {StormDuration} steps ago 
        # +/- the charging that occurs under the modified generation capacity   
        storageAdj = [
            np.lib.stride_tricks.sliding_window_view(
                np.concatenate([np.zeros(stormDur[i] - 1), #function only recognises full length windows -> pad with zeros
                                windDiff[:,i]]), 
                stormDur[i]).sum(axis=1) 
            for i in stormZone]
        
        if len(storageAdj) == 1: storageAdj = storageAdj[0]
        else: storageAdj = np.stack(storageAdj, axis = 1).sum(axis=1)
    
        Storage_1, StorageD_1 = np.roll(Storage, 1, axis = 0), np.roll(StorageD, 1, axis = 0)
        Storage_1[0], StorageD_1[0] = 0.5 * Scapacity, 0.5 * ScapacityD
        
        RStorage_1 = np.maximum(0, Storage_1 + storageAdj)
        RStorageD_1 = np.maximum(0, StorageD_1 + Storage_1 + storageAdj)
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
    else: 
        RCharge, RChargeD, RDischarge, RDischargeD, RP2V = Charge, ChargeD, Discharge, DischargeD, P2V
        RStorage, RStorageD, RDeficit, RDeficitD, RSpillage = Storage, StorageD, Deficit, DeficitD, Spillage

    Deficit = np.maximum(Netload - Discharge + P2V, 0)
    DeficitD = ConsumeD - DischargeD - P2V * efficiencyD
    Spillage = -1 * np.minimum(Netload + Charge + ChargeD, 0)

    
    RDeficit = np.maximum(RNetload - RDischarge + RP2V, 0)
    RDeficitD = ConsumeD - RDischargeD - RP2V * efficiencyD
    RSpillage = -1 * np.minimum(RNetload + RCharge + RChargeD, 0)

    assert 0 <= int(np.amax(Storage)) <= Scapacity, 'Storage below zero or exceeds max storage capacity'
    assert 0 <= int(np.amax(StorageD)) <= ScapacityD, 'StorageD below zero or exceeds max storage capacity'
    assert np.amin(Deficit) >= 0, 'Deficit below zero'
    assert np.amin(DeficitD) > -0.1, 'DeficitD below zero: {}'.format(np.amin(DeficitD))
    assert np.amin(Spillage) >= 0, 'Spillage below zero'
    assert np.amin(RDeficit) >= 0, 'RDeficit below zero'
    assert np.amin(RDeficitD) > -0.1, 'RDeficitD below zero: {}'.format(np.amin(RDeficitD))
    assert np.amin(RSpillage) >= 0, 'RSpillage below zero'

    solution.Discharge, solution.Charge, solution.Storage, solution.P2V = (Discharge, Charge, Storage, P2V)
    solution.DischargeD, solution.ChargeD, solution.StorageD = (DischargeD, ChargeD, StorageD)
    solution.Deficit, solution.DeficitD, solution.Spillage = (Deficit, DeficitD, Spillage)

    solution.RDischarge, solution.RCharge, solution.RStorage, solution.RP2V = (RDischarge, RCharge, RStorage, RP2V)
    solution.RDischargeD, solution.RChargeD, solution.RStorageD = (RDischargeD, RChargeD, RStorageD)
    solution.RDeficit, solution.RDeficitD, solution.RSpillage = (RDeficit, RDeficitD, RSpillage) 
    # solution.Surplus, solution.SurplusD = (Surplus, SurplusD)

    return Deficit, DeficitD, RDeficit, RDeficitD#, Surplus, SurplusD