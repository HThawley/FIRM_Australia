# To simulate energy supply-demand balance based on long-term, high-resolution chronological data
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

import numpy as np


def Resilience(solution, flexible, start=None, end=None):
    """Deficit = Simulation.Reliability(S, hydro=...)"""

    Netload = (solution.MLoad.sum(axis=1) - solution.GPV.sum(axis=1) - solution.GWind.sum(axis=1) - solution.GBaseload.sum(axis=1))[start:end] - flexible # Sj-ENLoad(j, t), MW

    windDiff, stormDur, stormZone = solution.WindDiff, solution.stormDur, solution.stormZone

    # if stormZone.shape:
    RNetload = Netload + windDiff.sum(axis=1) 
    # else: 
    #   RNetload = Netload + windDiff.flatten()

    length = len(Netload)
    solution.flexible = flexible # MW

    Pcapacity = sum(solution.CPHP) * pow(10, 3) # S-CPHP(j), GW to MW
    Scapacity = solution.CPHS * pow(10, 3) # S-CPHS(j), GWh to MWh
    PcapacityD = sum(solution.CDP) * pow(10, 3) # S-CDP(j), GW to MW
    ScapacityD = sum(solution.CDS) * pow(10, 3) # S-CDS(j), GWh to MWh
    efficiency, efficiencyD, resolution = (solution.efficiency, solution.efficiencyD, solution.resolution)

    Discharge, Charge, Storage, DischargeD, ChargeD, StorageD, P2V = map(np.zeros, [length] * 7)
    # Surplus, SurplusD = map(np.zeros, [length]*2)
    RDischarge, RCharge, RStorage, RDischargeD, RChargeD, RStorageD, RP2V = map(np.zeros, [length]*7)
    ConsumeD = solution.MLoadD.sum(axis=1)[start:end] * efficiencyD

    for t in range(100):

        Netloadt = Netload[t]
        Storaget_1 = Storage[t-1] if t>0 else 0.5 * Scapacity
        StorageDt_1 = StorageD[t-1] if t>0 else 0.5 * ScapacityD

        Discharget = min(max(0, Netloadt), Pcapacity, Storaget_1 / resolution)
        Charget = min(-1 * min(0, Netloadt), Pcapacity, (Scapacity - Storaget_1) / efficiency / resolution)
        Storaget = Storaget_1 - Discharget * resolution + Charget * resolution * efficiency
        
        ConsumeDt = ConsumeD[t]

        DischargeDt = min(ConsumeDt, StorageDt_1 / resolution)
        ChargeDt = min(-1 * min(0, Netloadt + Charget), PcapacityD, (ScapacityD - StorageDt_1) / efficiencyD / resolution)
        StorageDt = StorageDt_1 - DischargeDt * resolution + ChargeDt * resolution * efficiencyD

        diff = ConsumeDt - DischargeDt
        P2Vt = min(diff / efficiencyD, Pcapacity - Discharget - Charget) if diff > 0 and Storaget / resolution > diff / efficiencyD else 0

        Discharge[t] = Discharget + P2Vt
        P2V[t] = P2Vt
        Charge[t] = Charget
        Storage[t] = Storaget - P2Vt * resolution        

        DischargeD[t] = DischargeDt
        ChargeD[t] = ChargeDt
        StorageD[t] = StorageDt
        
        # Calculate surplus
        # Surplus[t] = max(0, -1*min(0,Netloadt) - Charget) * resolution #MWh
        # SurplusD[t] = max(0, min(0,-1 * min(0, Netloadt + Charget) - ChargeDt)) * resolution 

# =============================================================================
#         # Re-simulate with resilience losses due to windstorm, and including storage depletion
# =============================================================================
    if len(stormZone) > 0:    
        # State of charge is taken as the state of charge {StormDuration} steps ago 
        # +/- the charging that occurs under the modified generation capacity   
        storageAdj = [
            np.lib.stride_tricks.sliding_window_view(
                np.concatenate([np.zeros(stormDur[i] - 1), #function only recognises full length windows -> pad with zeros
                                windDiff[:,i]]), 
                stormDur[i]).sum(axis=1) 
            for i in stormZone]
        if len(storageAdj) == 1: 
            storageAdj = storageAdj[0]
        elif len(storageAdj) > 1:
            storageAdj = np.stack(storageAdj, axis = 1).sum(axis=1)
    
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
      

    Deficit = np.maximum(Netload - Discharge + P2V, 0)
    DeficitD = ConsumeD - DischargeD - P2V * efficiencyD
    Spillage = -1 * np.minimum(Netload + Charge + ChargeD, 0)

    
    RDeficit = np.maximum(RNetload - RDischarge + RP2V, 0)
    RDeficitD = ConsumeD - DischargeD - RP2V * efficiencyD
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