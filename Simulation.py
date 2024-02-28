# To simulate energy supply-demand balance based on long-term, high-resolution chronological data
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

import numpy as np

def Reliability(solution, flexible, start=None, end=None):
    """Deficit = Simulation.Reliability(S, hydro=...)"""

    Netload = (solution.MLoad.sum(axis=1) - solution.GPV.sum(axis=1) - solution.GWind.sum(axis=1) - solution.GBaseload.sum(axis=1))[start:end] \
              - flexible # Sj-ENLoad(j, t), MW

    length, nvec = Netload.shape
    solution.flexible = flexible # MW

    Pcapacity = sum(solution.CPHP) * pow(10, 3) # S-CPHP(j), GW to MW
    Scapacity = solution.CPHS * pow(10, 3) # S-CPHS(j), GWh to MWh
    PcapacityD = sum(solution.CDP) * pow(10, 3) # S-CDP(j), GW to MW
    ScapacityD = sum(solution.CDS) * pow(10, 3) # S-CDS(j), GWh to MWh
    efficiency, efficiencyD, resolution = (solution.efficiency, solution.efficiencyD, solution.resolution)

    Discharge, Charge, Storage, DischargeD, ChargeD, StorageD, P2V = map(np.zeros, [(length, nvec)] * 7)
    ConsumeD = np.stack([solution.MLoadD.sum(axis=(1,2))[start:end] * efficiencyD]*nvec).T

    for t in range(length):

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
        P2Vt = np.zeros(nvec) 
        diffMask = (diff > 0) * (Storaget / resolution > diff / efficiencyD )
        P2Vt[diffMask] = np.minimum(diff / efficiencyD, Pcapacity - Discharget - Charget)[diffMask]

        Discharge[t] = Discharget + P2Vt
        P2V[t] = P2Vt
        Charge[t] = Charget
        Storage[t] = Storaget - P2Vt * resolution

        DischargeD[t] = DischargeDt
        ChargeD[t] = ChargeDt
        StorageD[t] = StorageDt

    Deficit = np.maximum(Netload - Discharge + P2V, 0)
    DeficitD = ConsumeD - DischargeD - P2V * efficiencyD
    Spillage = -1 * np.clip(Netload + Charge + ChargeD, None, 0)

    assert (0 <= np.floor(np.amax(Storage, axis=0))).all(), 'Storage below zero '
    assert (np.floor(np.amax(Storage, axis=0)) <= Scapacity).all(), 'Storage exceeds max storage capacity'
    assert (0 <= np.floor(np.amax(StorageD, axis=0))).all(), 'StorageD below zero'
    assert (np.floor(np.amax(StorageD, axis=0)) <= ScapacityD).all(), 'StorageD or exceeds max storage capacity'
    assert np.amin(Deficit) >= 0, 'Deficit below zero'
    assert np.amin(DeficitD) > -0.1, 'DeficitD below zero: {}'.format(np.amin(DeficitD))
    assert np.amin(Spillage) >= 0, 'Spillage below zero'

    solution.Discharge, solution.Charge, solution.Storage, solution.P2V = (Discharge, Charge, Storage, P2V)
    solution.DischargeD, solution.ChargeD, solution.StorageD = (DischargeD, ChargeD, StorageD)
    solution.Deficit, solution.DeficitD, solution.Spillage = (Deficit, DeficitD, Spillage)

    return Deficit, DeficitD