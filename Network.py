# A transmission network model to calculate inter-regional power flows
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

import numpy as np

def Transmission(solution, output=False):
    """TDC = Network.Transmission(S)"""

    Nodel, PVl, OnsWl, OffsWl = solution.Nodel, solution.PVl, solution.OnsWl, solution.OffsWl
    intervals, nodes = solution.intervals, solution.nodes

    MPV, MOnsW, MOffsW = map(np.zeros, [(nodes, intervals)] * 3)
    for i, j in enumerate(Nodel):
        MPV[i,    :] = solution.GPV[:,    np.where(PVl   ==j)[0]].sum(axis=1)
        MOnsW[i,  :] = solution.GOnsW[:,  np.where(OnsWl ==j)[0]].sum(axis=1)
        MOffsW[i, :] = solution.GOffsW[:, np.where(OffsWl==j)[0]].sum(axis=1)
    MPV, MOnsW, MOffsW = MPV.T, MOnsW.T, MOffsW.T # Sij-GPV(t, i), Sij-GWind(t, i), MW

    MLoad, MBaseload, CPeak = solution.MLoad, solution.GBaseload, solution.CPeak # MW, MW, GW

    pkfactor = np.tile(CPeak, (intervals, 1)) / CPeak.sum()
    MPeak = np.tile(solution.flexible, (nodes, 1)).T * pkfactor # MW

    MLoad_pos = MLoad - MLoad.min()
    defactor = np.divide(MLoad_pos, 1e-12 + MLoad_pos.sum(axis=1)[:, None])
    MDeficit = np.tile(solution.Deficit, (nodes, 1)).T * defactor # MDeficit: EDE(j, t)

    MPW = MPV + MOnsW + MOffsW
    spfactor = np.divide(MPW, MPW.sum(axis=1)[:, None], where=MPW.sum(axis=1)[:, None]!=0)
    MSpillage = np.tile(solution.Spillage, (nodes, 1)).T * spfactor # MSpillage: ESP(j, t)

    CPHP = solution.CPHP
    pcfactor = CPHP / sum(CPHP) if sum(CPHP) != 0 else 0
    MDischarge = np.tile(solution.Discharge, (nodes, 1)).T * pcfactor # MDischarge: DPH(j, t)
    MCharge = np.tile(solution.Charge, (nodes, 1)).T * pcfactor # MCharge: CHPH(j, t)

    MImport = (MLoad + MCharge + MSpillage 
              - MPV - MOnsW - MOffsW - MBaseload - MPeak - MDischarge - MDeficit) # EIM(t, j), MW

    FQ = -MImport[:, np.where(Nodel=='FNQ')[0][0]] if 'FNQ' in Nodel else np.zeros(intervals)
    AS = -MImport[:, np.where(Nodel=='NT' )[0][0]] if 'NT'  in Nodel else np.zeros(intervals)
    SW =  MImport[:, np.where(Nodel=='WA' )[0][0]] if 'WA'  in Nodel else np.zeros(intervals)
    TV = -MImport[:, np.where(Nodel=='TAS')[0][0]]
    NQ =  MImport[:, np.where(Nodel=='QLD')[0][0]] - FQ
    NV =  MImport[:, np.where(Nodel=='VIC')[0][0]] - TV
    NS = -MImport[:, np.where(Nodel=='NSW')[0][0]] - NQ - NV
    NS1 = MImport[:, np.where(Nodel=='SA' )[0][0]] - AS + SW
    assert abs(NS - NS1).max()<=0.1, abs(NS - NS1).max()
    
    TDC = np.array([FQ, NQ, NS, NV, AS, SW, TV]).T # TDC(t, k), MW

    if output:
        MStorage = np.tile(solution.Storage, (nodes, 1)).transpose() * pcfactor # SPH(t, j), MWh
        solution.MPV, solution.MOnsW, solution.MOffsW, solution.MBaseload, solution.MPeak = (MPV, MOnsW, MOffsW, MBaseload, MPeak)
        solution.MDischarge, solution.MCharge, solution.MStorage = MDischarge, MCharge, MStorage
        solution.MDeficit, solution.MSpillage = (MDeficit, MSpillage)

    return TDC