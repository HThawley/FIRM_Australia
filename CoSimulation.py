# To simulate energy supply-demand balance based on long-term, high-resolution chronological data
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

import numpy as np
from Simulation import Reliability

def Resilience(solution, flexible, start=None, end=None, output = 'deficits'):
    """ """

    windDiff, eventDur, eventZoneIndx = solution.WindDiff[start:end], solution.eventDur, solution.eventZoneIndx
    
    solution.flexible = flexible # MW

    Pcapacity = sum(solution.CPHP) * pow(10, 3) # S-CPHP(j), GW to MW
    Scapacity = solution.CPHS * pow(10, 3) # S-CPHS(j), GWh to MWh
    PcapacityD = sum(solution.CDP) * pow(10, 3) # S-CDP(j), GW to MW
    ScapacityD = sum(solution.CDS) * pow(10, 3) # S-CDS(j), GWh to MWh
    efficiency, efficiencyD, resolution = (solution.efficiency, solution.efficiencyD, solution.resolution)

    ConsumeD = solution.MLoadD.sum(axis=1)[start:end] * efficiencyD

    (solution.RDischarge, solution.RCharge, solution.RStorage, solution.RP2V, 
     solution.RDischargeD, solution.RChargeD, solution.RStorageD, solution.RDeficit, 
     solution.RDeficitD, solution.RSpillage) = map(np.zeros, [windDiff.shape] * 10)

            
    (Netload, Storage, StorageD, Deficit, DeficitD, Spillage, Discharge, DischargeD, 
         Charge, ChargeD, P2V) = Reliability(solution, flexible, start, end, output=False, resilience=True)
    
    def simulate_resilience_losses(i=None):
        """Re-simulate with resilience losses due to windevent, and including storage depletion"""
        Storage_1, StorageD_1 = np.roll(Storage, 1, axis = 0), np.roll(StorageD, 1, axis = 0)
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
    
        assert 0 <= int(np.amax(RStorage)) <= Scapacity, 'RStorage below zero or exceeds max storage capacity'
        assert 0 <= int(np.amax(RStorageD)) <= ScapacityD, 'RStorageD below zero or exceeds max storage capacity'
        assert np.amin(RDeficit) >= 0, 'RDeficit below zero'
        assert np.amin(RDeficitD) > -0.1, 'RDeficitD below zero: {}'.format(np.amin(RDeficitD))
        assert np.amin(RSpillage) >= 0, 'RSpillage below zero'
        
        if i is not None: 
            solution.RDischarge[:,i], solution.RCharge[:,i], solution.RStorage[:,i], solution.RP2V[:,i] = (RDischarge, RCharge, RStorage, RP2V)
            solution.RDischargeD[:,i], solution.RChargeD[:,i], solution.RStorageD[:,i] = (RDischargeD, RChargeD, RStorageD)
            solution.RDeficit[:,i], solution.RDeficitD[:,i], solution.RSpillage[:,i] = (RDeficit, RDeficitD, RSpillage)                 
        else: 
            solution.RDischarge, solution.RCharge, solution.RStorage, solution.RP2V = (RDischarge, RCharge, RStorage, RP2V)
            solution.RDischargeD, solution.RChargeD, solution.RStorageD = (RDischargeD, RChargeD, RStorageD)
            solution.RDeficit, solution.RDeficitD, solution.RSpillage = (RDeficit, RDeficitD, RSpillage) 
        
    
    if eventZoneIndx is not None: 
        if len(eventZoneIndx) > 1 and solution.logic == 'and':
            eventDur = np.unique(eventDur[np.where(eventDur!=0)])
            assert len(eventDur) == 1
            eventZoneIndx = np.array([-1])
            windDiff = windDiff.sum(axis=1).reshape((-1,1))
        
        if len(eventZoneIndx) == 1: 
            RNetload = Netload - windDiff.sum(axis=1)
    
            # State of charge is taken as the state of charge {eventDuration} steps ago 
            # +/- the charging that occurs under the modified generation capacity   
            storageAdj = [
                np.lib.stride_tricks.sliding_window_view(
                    np.concatenate([np.zeros(eventDur[i] - 1), #function only recognises full length windows -> pad with zeros
                                    np.clip(windDiff[:,i]-Spillage, None, 0)]), 
                    eventDur[i]).sum(axis=1)
            for i in eventZoneIndx]
                
            if len(storageAdj) == 1: 
                storageAdj = storageAdj[0] 
            else: 
                storageAdj = np.stack(storageAdj, axis = 1).sum(axis=1)
    
            simulate_resilience_losses(i=None)
            
        
        if len(eventZoneIndx) > 1:
            assert solution.logic == 'xor'
            
            for i in eventZoneIndx:
                
                storageAdj = np.lib.stride_tricks.sliding_window_view(
                        np.concatenate([np.zeros(eventDur[i] - 1), #function only recognises full length windows -> pad with zeros
                                        np.clip(windDiff[:,i]-Spillage, None, 0)]), 
                        eventDur[i]).sum(axis=1)
                
                RNetload = Netload - windDiff[:,i], 
            
                simulate_resilience_losses(i)
                
    else: 
        RCharge, RChargeD, RDischarge, RDischargeD, RP2V, RStorage, RStorageD, RDeficit, RDeficitD, RSpillage = (
            Charge, ChargeD, Discharge, DischargeD, P2V, Storage, StorageD, Deficit, DeficitD, Spillage)
        storageAdj = np.zeros(Netload.shape)
        RNetload = Netload.copy()
        simulate_resilience_losses()
        
    solution.Discharge, solution.Charge, solution.Storage, solution.P2V = (Discharge, Charge, Storage, P2V)
    solution.DischargeD, solution.ChargeD, solution.StorageD = (DischargeD, ChargeD, StorageD)
    solution.Deficit, solution.DeficitD, solution.Spillage = (Deficit, DeficitD, Spillage)

    if output =='deficits': 
        return Deficit, DeficitD, solution.RDeficit, solution.RDeficitD
    elif output == 'solution': 
        return solution
    elif output is None: 
        return None
    else: 
        raise Exception("Specify output")