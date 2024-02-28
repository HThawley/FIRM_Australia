# A transmission network model to calculate inter-regional power flows
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

import numpy as np

def Transmission(solution, output=False):
    """TDC = Network.Transmission(S)"""

    Nodel, PVl, Windl = (solution.Nodel, solution.PVl, solution.Windl)
    intervals, nodes, nvec = (solution.intervals, solution.nodes, solution.nvec)

    MPV, MWind = map(np.zeros, [(intervals, nodes, nvec)] * 2)
    for i, j in enumerate(Nodel):
        MPV[:, i, :] = solution.GPV[:, np.where(PVl==j)[0], :].sum(axis=1)
        MWind[:, i, :] = solution.GWind[:, np.where(Windl==j)[0], :].sum(axis=1)

    MBaseload = solution.GBaseload # MW
    CPeak = solution.CPeak # GW
    pkfactor = np.tile(CPeak, (intervals, 1)) / CPeak.sum()
    MPeak = np.tile(solution.flexible.T, (1, nodes, 1)).T * pkfactor.reshape(intervals, nodes, 1) # MW

    MLoad = solution.MLoad # EOLoad(t, j), MW

    defactor = MLoad / MLoad.sum(axis=(1)).reshape(-1, 1, 1)
    MDeficit = np.tile(solution.Deficit, (nodes, 1, 1)).transpose((1,0,2)) * defactor # MDeficit: EDE(j, t)

    MPW = MPV + MWind
    spfactor = np.divide(MPW, MPW.sum(axis=1)[:, None], where=MPW.sum(axis=1)[:, None]!=0)    
    
    MSpillage = np.tile(solution.Spillage, (nodes, 1, 1)).transpose((1,0,2)) * spfactor # MSpillage: ESP(j, t)

    CPHP = solution.CPHP
    pcfactor = np.zeros((intervals, nodes, nvec))
    dzsm = CPHP.sum(axis=0) != 0 # divide by zero safe mask
    pcfactor[:,:,dzsm] = np.tile(CPHP[:,dzsm], (intervals, 1, 1)) / CPHP[:,dzsm].sum(axis=0)
    
    MDischarge = np.tile(solution.Discharge, (nodes, 1, 1)).transpose(1,0,2) * pcfactor # MDischarge: DPH(j, t)
    MCharge = np.tile(solution.Charge, (nodes, 1, 1)).transpose(1,0,2) * pcfactor # MCharge: CHPH(j, t)

    MP2V = np.tile(solution.P2V, (nodes, 1, 1)).transpose(1,0,2) * pcfactor # MP2V: DP2V(j, t)

    CDP = solution.CDP
    pcfactorD = np.tile(CDP, (intervals, 1)) / sum(CDP) if sum(CDP) != 0 else np.zeros((1, 1))
    MChargeD = np.tile(solution.ChargeD, (nodes, 1, 1)).transpose(1,0,2) * pcfactorD.reshape(*pcfactorD.shape, 1) # MChargeD: CHD(j, t)

    MImport = MLoad + MCharge + MChargeD + MSpillage \
              - MPV - MWind - MBaseload - MPeak - MDischarge + MP2V - MDeficit # EIM(t, j), MW

    FQ = -1 * MImport[:, np.where(Nodel=='FNQ')[0][0], :] if 'FNQ' in Nodel else np.zeros((intervals, nvec))
    AS = -1 * MImport[:, np.where(Nodel=='NT')[0][0], :] if 'NT' in Nodel else np.zeros((intervals, nvec))
    SW = MImport[:, np.where(Nodel=='WA')[0][0], :] if 'WA' in Nodel else np.zeros((intervals, nvec))
    TV = -1 * MImport[:, np.where(Nodel=='TAS')[0][0], :]

    NQ = MImport[:, np.where(Nodel=='QLD')[0][0], :] - FQ
    NV = MImport[:, np.where(Nodel=='VIC')[0][0], :] - TV

    NS = -1 * MImport[:, np.where(Nodel=='NSW')[0][0], :] - NQ - NV
    NS1 = MImport[:, np.where(Nodel=='SA')[0][0], :] - AS + SW
    assert np.abs(NS - NS1).max()<=0.1, print(np.abs(NS - NS1).max())

    TDC = np.array([FQ, NQ, NS, NV, AS, SW, TV]).transpose(1,0,2) # TDC(t, k), MW

    if output:
        MStorage = np.tile(solution.Storage, (nodes, 1, 1)).transpose((1,0,2)) * pcfactor # SPH(t, j), MWh
        MDischargeD = np.tile(solution.DischargeD, (nodes, 1, 1)).transpose((1,0,2)) * pcfactorD  # MDischargeD: DD(j, t)
        MStorageD = np.tile(solution.StorageD, (nodes, 1, 1)).transpose((1,0,2)) * pcfactorD  # SD(t, j), MWh
        solution.MPV, solution.MWind, solution.MBaseload, solution.MPeak = (MPV, MWind,MBaseload, MPeak)
        solution.MDischarge, solution.MCharge, solution.MStorage, solution.MP2V = (MDischarge, MCharge, MStorage, MP2V)
        solution.MDischargeD, solution.MChargeD, solution.MStorageD = (MDischargeD, MChargeD, MStorageD)
        solution.MDeficit, solution.MSpillage = (MDeficit, MSpillage)

    return TDC