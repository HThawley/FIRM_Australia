# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 09:05:54 2023

@author: hmtha
"""


import numpy as np
from Simulation import Reliability

def DeficitSimulation(solution, flexible, RSim, start=None, end=None, output = 'deficits'):
    """Deficit = Simulation.Resilience(S, hydro=...)"""

    windDiff, eventDur, eventZoneIndx = solution.WindDiff[start:end], solution.eventDur, solution.eventZoneIndx
    assert eventZoneIndx is not None
    
    solution.flexible = flexible # MW

    Pcapacity = sum(solution.CPHP) * pow(10, 3) # S-CPHP(j), GW to MW
    Scapacity = solution.CPHS * pow(10, 3) # S-CPHS(j), GWh to MWh
    PcapacityD = sum(solution.CDP) * pow(10, 3) # S-CDP(j), GW to MW
    ScapacityD = sum(solution.CDS) * pow(10, 3) # S-CDS(j), GWh to MWh
    efficiency, efficiencyD, resolution = (solution.efficiency, solution.efficiencyD, solution.resolution)

    ConsumeD = solution.MLoadD.sum(axis=1)[start:end] * efficiencyD

    (Netload, Storage, StorageD, Deficit, DeficitD, Spillage, Discharge, DischargeD, 
         Charge, ChargeD, P2V) = Reliability(solution, flexible, output=False, resilience = True)
 
  
    # Calculate power generation difference for duration of event
    windDiffInst = np.zeros(windDiff.shape)
    for i in eventZoneIndx:
        windDiffInst[max(0, RSim-eventDur[i]):RSim+1, i] = windDiff[max(0, RSim-eventDur[i]):RSim+1, i] 
    
    # Create new net load 
    try: 
        RNetload = Netload - windDiffInst.sum(axis=1) 
    except np.AxisError: 
        RNetload = Netload - windDiffInst
    
    
    RStorage, RStorageD = Storage.copy(), StorageD.copy()
    RDischarge, RDischargeD = Discharge.copy(), DischargeD.copy()
    RCharge, RChargeD =Charge.copy(), ChargeD.copy()
    RP2V = P2V.copy()

    # Recalculate step-wise energy flows from start of event 
    for t in range(RSim-eventDur[i], solution.intervals):
        RNetloadt = RNetload[t]
        
        RStoraget_1 = RStorage[t-1] if t>0 else 0.5 * Scapacity
        RStorageDt_1 = RStorageD[t-1] if t>0 else 0.5 * ScapacityD

        RDischarget = min(max(0, RNetloadt), Pcapacity, RStoraget_1 / resolution)
        RCharget = min(-1 * min(0, RNetloadt), Pcapacity, (Scapacity - RStoraget_1) / efficiency / resolution)
        RStoraget = RStoraget_1 - RDischarget * resolution + RCharget * resolution * efficiency
        
        ConsumeDt = ConsumeD[t]

        RDischargeDt = min(ConsumeDt, RStorageDt_1 / resolution)
        RChargeDt = min(-1 * min(0, RNetloadt + RCharget), PcapacityD, (ScapacityD - RStorageDt_1) / efficiencyD / resolution)
        RStorageDt = RStorageDt_1 - RDischargeDt * resolution + RChargeDt * resolution * efficiencyD

        Rdiff = ConsumeDt - RDischargeDt
        RP2Vt = min(Rdiff / efficiencyD, Pcapacity - RDischarget - RCharget) if Rdiff > 0 and RStoraget / resolution > Rdiff / efficiencyD else 0

        RDischarge[t] = RDischarget + RP2Vt
        RP2V[t] = RP2Vt
        RCharge[t] = RCharget
        RStorage[t] = RStoraget - RP2Vt * resolution        

        RDischargeD[t] = RDischargeDt
        RChargeD[t] = RChargeDt
        RStorageD[t] = RStorageDt

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

    # solution.RDischarge, solution.RCharge, solution.RStorage, solution.RP2V = (RDischarge, RCharge, RStorage, RP2V)
    # solution.RDischargeD, solution.RChargeD, solution.RStorageD = (RDischargeD, RChargeD, RStorageD)
    # solution.RDeficit, solution.RDeficitD, solution.RSpillage = (RDeficit, RDeficitD, RSpillage) 

    if output =='deficits': 
        return Deficit, DeficitD, RDeficit, RDeficitD
    elif output == 'solution': 
        return solution
    elif output is None: 
        return None
    else: 
        raise Exception("Specify output")