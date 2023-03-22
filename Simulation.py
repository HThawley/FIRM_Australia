# To simulate energy supply-demand balance based on long-term, high-resolution chronological data
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

import numpy as np

def Reliability(solution, flexible, start=None, end=None):
    """Deficit = Simulation.Reliability(S, hydro=...)"""

    Netload = (solution.MLoad.sum(axis=1) - solution.GPV.sum(axis=1) - solution.GWind.sum(axis=1) - solution.GBaseload.sum(axis=1))[start:end] \
              - flexible # Sj-ENLoad(j, t), MW

    RNetload = (solution.MLoad.sum(axis=1) - solution.GPV.sum(axis=1) - solution.GWindR.sum(axis=1) - solution.GBaseload.sum(axis=1))[start:end] \
              - flexible # Sj-ENLoad(j, t), MW

    length = len(Netload)
    solution.flexible = flexible # MW

    Pcapacity = sum(solution.CPHP) * pow(10, 3) # S-CPHP(j), GW to MW
    Scapacity = solution.CPHS * pow(10, 3) # S-CPHS(j), GWh to MWh
    PcapacityD = sum(solution.CDP) * pow(10, 3) # S-CDP(j), GW to MW
    ScapacityD = sum(solution.CDS) * pow(10, 3) # S-CDS(j), GWh to MWh
    efficiency, efficiencyD, resolution = (solution.efficiency, solution.efficiencyD, solution.resolution)

    Discharge, Charge, Storage, DischargeD, ChargeD, StorageD, P2V, Surplus, SurplusD = map(np.zeros, [length] * 9)
    RDischarge, RCharge, RStorage, RDischargeD, RChargeD, RStorageD, RP2V = map(np.zeros, [length]*7)
    ConsumeD = solution.MLoadD.sum(axis=1)[start:end] * efficiencyD

    for t in range(length):

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
        Surplus[t] = max(0, -1*min(0,Netloadt) - Charget)
        SurplusD[t] = max(0, min(0,-1 * min(0, Netloadt + Charget) - ChargeDt))

        # Re-simulate with resilience losses due to windstorm, and including storage depletion
        StormDuration = 2 *24*2 
        #2 days*24 hours per day *2 half hours per hour 
        # This parameter will be moved elsewhere, kept here now for testing
        
        RNetloadt = RNetload[t]

        period = range(min(0, t-StormDuration), t)
        RStoraget_1 = Storage[min(0, t-StormDuration)] \
                + RCharge[period].sum() * resolution * efficiency \
                - RDischarge[period].sum() * resolution \
                if t>0 else 0.5*Scapacity
                
        RStorageDt_1 = StorageD[min(0, t-StormDuration)] \
                + RCharge[period].sum() * resolution * efficiencyD \
                - RDischarge[period].sum() * resolution \
                if t>0 else 0.5*Scapacity
        # State of charge is taken as the state of charge {StormDuration} steps ago 
        # +/- the charging that occurs under the modified generation capacity 
        
        RDischarget = min(max(0, RNetloadt), Pcapacity, RStoraget_1 / resolution)
        RCharget = min(-1 * min(0, RNetloadt), Pcapacity, (Scapacity - RStoraget_1) / efficiency / resolution)
        RStoraget = RStoraget_1 - RDischarget * resolution + RCharget * resolution * efficiency
    
        RDischargeDt = min(ConsumeDt, StorageDt_1 / resolution)
        RChargeDt = min(-1 * min(0, RNetloadt + Charget), PcapacityD, (ScapacityD - StorageDt_1) / efficiencyD / resolution)
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
    

        
        

        '''# Re-simulate with resilience losses due to windstorm
        RNetloadt = RNetload[t]

        RDischarget = min(max(0, RNetloadt), Pcapacity, Storaget_1 / resolution)
        RCharget = min(-1 * min(0, RNetloadt), Pcapacity, (Scapacity - Storaget_1) / efficiency / resolution)
        RStoraget = Storaget_1 - RDischarget * resolution + RCharget * resolution * efficiency

        RChargeDt = min(-1 * min(0, RNetloadt + RCharget), PcapacityD, (ScapacityD - StorageDt_1) / efficiencyD / resolution)
        RStorageDt = StorageDt_1 - DischargeDt * resolution + RChargeDt * resolution * efficiencyD

        RP2Vt = min(diff / efficiencyD, Pcapacity - RDischarget - RCharget) if diff > 0 and RStoraget / resolution > diff / efficiencyD else 0

        RDischarge[t] = RDischarget + RP2Vt
        RP2V[t] = RP2Vt
        RCharge[t] = RCharget
        RStorage[t] = RStoraget - RP2Vt * resolution        

        RChargeD[t] = RChargeDt
        RStorageD[t] = RStorageDt'''

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
    solution.RDeficit, solution.RDeficitD, solution.Spillage = (RDeficit, RDeficitD, RSpillage)
    solution.Surplus, solution.SurplusD = (Surplus, SurplusD)

    return Deficit, DeficitD, RDeficit, RDeficitD, Surplus, SurplusD