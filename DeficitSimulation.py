# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 09:05:54 2023

@author: hmtha
"""
##TODO 
# Alter hydro and bio here
# Correct charging behaviour

import numpy as np
from Simulation import Reliability

def DeficitSimulation(solution, flexible, FCap, RSim, start=None, end=None, output=True):
    """Deficit = Simulation.Resilience(S, hydro=...)"""
    Netload = (solution.MLoad.sum(axis=1) - solution.GPV.sum(axis=1) - solution.GWind.sum(axis=1) - solution.GBaseload.sum(axis=1))[start:end] # Sj-ENLoad(j, t), MW
    length = len(Netload)

    windDiff, eventDur, eventZoneIndx = solution.WindDiff[start:end], solution.eventDur, solution.eventZoneIndx
    assert eventZoneIndx is not None
    

    Pcapacity = sum(solution.CPHP) * 1000 # S-CPHP(j), GW to MW
    Scapacity = solution.CPHS * 1000 # S-CPHS(j), GWh to MWh
    PcapacityD = sum(solution.CDP) * 1000 # S-CDP(j), GW to MW
    ScapacityD = sum(solution.CDS) * 1000 # S-CDS(j), GWh to MWh
    efficiency, efficiencyD, resolution = (solution.efficiency, solution.efficiencyD, solution.resolution)

    ConsumeD = solution.MLoadD.sum(axis=1)[start:end] * efficiencyD

    Deficit, DeficitD = Reliability(solution, flexible, start, end, output=True)
  
    # Calculate power generation difference for duration of event
    windDiffInst = np.zeros(windDiff.shape)
    for i in eventZoneIndx:
        windDiffInst[max(0, RSim-eventDur):RSim+1, i] = windDiff[max(0, RSim-eventDur):RSim+1, i] 
    
    # Create new netload 
    try: 
        RNetload = Netload - windDiffInst.sum(axis=1) - flexible
    except np.AxisError: 
        RNetload = Netload - windDiffInst
        
    RStorage, RStorageD = solution.Storage.copy(), solution.StorageD.copy()
    RDischarge, RDischargeD = solution.Discharge.copy(), solution.DischargeD.copy()
    RCharge, RChargeD = solution.Charge.copy(), solution.ChargeD.copy()
    RP2V = solution.P2V.copy()

    # Recalculate step-wise energy flows from start of event 
    for t in range(max(RSim-eventDur,0), min(length, RSim+eventDur*2)):
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

    from Dispatch import Flexible 
    RFlexible = solution.flexible.copy()
    startidx, endidx = max(0, RSim-eventDur*4), min(length, RSim+eventDur*2 )
    RFlexible[startidx:endidx] = Flexible((startidx, endidx), solution.x)
    

    assert 0 <= int(np.amax(RStorage)) <= Scapacity, 'Storage below zero or exceeds max storage capacity'
    assert 0 <= int(np.amax(RStorageD)) <= ScapacityD, 'StorageD below zero or exceeds max storage capacity'
    assert np.amin(RDeficit) >= 0, 'RDeficit below zero'
    assert np.amin(RDeficitD) > -0.1, 'RDeficitD below zero: {}'.format(np.amin(RDeficitD))
    assert np.amin(RSpillage) >= 0, 'RSpillage below zero'

    solution.RDischarge, solution.RCharge, solution.RStorage, solution.RP2V = (
        RDischarge.reshape(-1,1), RCharge.reshape(-1,1), RStorage.reshape(-1,1), RP2V.reshape(-1,1))
    solution.RDischargeD, solution.RChargeD, solution.RStorageD = (
        RDischargeD.reshape(-1,1), RChargeD.reshape(-1,1), RStorageD.reshape(-1,1))
    solution.RDeficit, solution.RDeficitD, solution.RSpillage = (
        RDeficit.reshape(-1,1), RDeficitD.reshape(-1,1), RSpillage.reshape(-1,1)) 
    solution.RFlexible = RFlexible

    return Deficit, DeficitD, RDeficit, RDeficitD

    