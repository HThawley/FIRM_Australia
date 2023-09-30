# Load profiles and generation mix data (LPGM) & energy generation, storage and transmission information (GGTA)
# based on x/capacities from Optimisation and flexible from Dispatch
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

from Input import *
from Simulation import Reliability
from CoSimulation import Resilience
from Network import Transmission

import numpy as np
import datetime as dt

def Debug(solution):
    """Debugging"""
    
    Load, PV, Wind = (solution.MLoad.sum(axis=1), solution.GPV.sum(axis=1), solution.GWind.sum(axis=1))
    Baseload, Peak = (solution.MBaseload.sum(axis=1), solution.MPeak.sum(axis=1))

    Discharge, Charge, Storage, P2V = (solution.Discharge, solution.Charge, solution.Storage, solution.P2V)
    DischargeD, ChargeD, StorageD = (solution.DischargeD, solution.ChargeD, solution.StorageD)
    Deficit, DeficitD, Spillage = (solution.Deficit, solution.DeficitD, solution.Spillage)

    PHS, DS = solution.CPHS * pow(10, 3), sum(solution.CDS) * pow(10, 3) # GWh to MWh
    efficiency, efficiencyD = solution.efficiency, solution.efficiencyD

    for i in range(intervals):
        # Energy supply-demand balance
        assert abs(Load[i] + Charge[i] + ChargeD[i] + Spillage[i]
                   - PV[i] - Wind[i] - Baseload[i] - Peak[i] - Discharge[i] + P2V[i] - Deficit[i]) <= 1

        # Discharge, Charge and Storage
        if i==0:
            assert abs(Storage[i] - 0.5 * PHS + Discharge[i] * resolution - Charge[i] * resolution * efficiency) <= 1
            assert abs(StorageD[i] - 0.5 * DS + DischargeD[i] * resolution - ChargeD[i] * resolution * efficiencyD) <= 1
        else:
            assert abs(Storage[i] - Storage[i - 1] + Discharge[i] * resolution - Charge[i] * resolution * efficiency) <= 1
            assert abs(StorageD[i] - StorageD[i - 1] + DischargeD[i] * resolution - ChargeD[i] * resolution * efficiencyD) <= 1

        # Capacity: PV, wind, Discharge, Charge and Storage
        # try:
        assert np.amax(PV) <= sum(solution.CPV) * pow(10, 3), print(np.amax(PV) - sum(solution.CPV) * pow(10, 3))
        assert np.amax(Wind) <= sum(solution.CWind) * pow(10, 3), print(np.amax(Wind) - sum(solution.CWind) * pow(10, 3))

        assert np.amax(Discharge) <= sum(solution.CPHP) * pow(10, 3), print(np.amax(Discharge) - sum(solution.CPHP) * pow(10, 3))
        assert np.amax(Charge) <= sum(solution.CPHP) * pow(10, 3), print(np.amax(Charge) - sum(solution.CPHP) * pow(10, 3))
        assert np.amax(Storage) <= solution.CPHS * pow(10, 3), print(np.amax(Storage) - solution.CPHS * pow(10, 3))
        assert np.amax(DischargeD) <= sum(solution.CDP) * pow(10, 3), print(np.amax(DischargeD) - sum(solution.CDP) * pow(10, 3))
        assert np.amax(ChargeD) <= sum(solution.CDP) * pow(10, 3), print(np.amax(ChargeD) - sum(solution.CDP) * pow(10, 3))
        assert np.amax(StorageD) <= sum(solution.CDS) * pow(10, 3), print(np.amax(StorageD) - sum(solution.CDS) * pow(10, 3))
        # except AssertionError:
        #     pass

    print('Debugging: everything is ok')

    return True

def LPGM(solution, RSim=None):
    """Load profiles and generation mix data"""

    C = np.stack([(solution.MLoad + solution.MLoadD).sum(axis=1), (solution.MLoad + solution.MChargeD + solution.MP2V).sum(axis=1),
                  solution.MHydro.sum(axis=1), solution.MBio.sum(axis=1), solution.GPV.sum(axis=1), solution.GWind.sum(axis=1), solution.GWindR.sum(axis=1),
                  solution.Discharge, solution.Deficit, -1 * solution.Spillage, -1 * solution.Charge, solution.Storage,
                  (solution.GWind - solution.GWindR).sum(axis=1), solution.RDeficit, solution.RStorage,solution.RDischarge, -1*solution.RCharge,-1*solution.RSpillage,
                  solution.FQ, solution.NQ, solution.NS, solution.NV, solution.AS, solution.SW, solution.TV])
    C = np.around(C.transpose())

    datentime = np.array([(dt.datetime(firstyear, 1, 1, 0, 0) + x * dt.timedelta(minutes=60 * resolution)).strftime('%a -%d %b %Y %H:%M') for x in range(intervals)])
    C = np.insert(C.astype('str'), 0, datentime, axis=1)

    header = 'Date & time,Operational demand (original),Operational demand (adjusted),' \
             'Hydropower,Biomass,Solar photovoltaics,Wind,StormPower,' \
             'PHES-power,Energy deficit,Energy spillage,PHES-Charge,PHES-Storage,' \
             'StormPowerLoss,StormDeficit,StormStorage,StormPHES-power,StormPHES-Charge,StormSpillage,' \
             'FNQ-QLD,NSW-QLD,NSW-SA,NSW-VIC,NT-SA,SA-WA,TAS-VIC'

    if RSim is None: np.savetxt('Results/S{}-{}-{}.csv'.format(scenario, stormZone, n_year), C, fmt='%s', delimiter=',', header=header, comments='')
    else: 
        C = C[max(0, RSim[1] - solution.stormDur.max() - 672):RSim[1] + 49, :] #2 weeks before storm + 1 day after 
        np.savetxt('Results/S-Deficit{}-{}-{}-{}.csv'.format(scenario, stormZone, n_year, RSim[0]), C, fmt='%s', delimiter=',', header=header, comments='')

    if scenario>=21:
        header = 'Date & time,Operational demand (original),Operational demand (adjusted),' \
                 'Hydropower,Biomass,Solar photovoltaics,Wind,StormPower,' \
                 'PHES-power,Energy deficit,Energy spillage,Transmission,PHES-Charge,' \
                 'PHES-Storage,StormPowerLoss,StormDeficit,'\
                 'StormStorage,StormPHES-power,StormPHES-charge,StormSpillage'

        Topology = solution.Topology[np.where(np.in1d(np.array(['FNQ', 'NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC', 'WA']), coverage) == True)[0]]

        for j in range(nodes):
            C = np.stack([(solution.MLoad + solution.MLoadD)[:, j], (solution.MLoad + solution.MChargeD + solution.MP2V)[:, j],
                          solution.MHydro[:, j], solution.MBio[:, j], solution.MPV[:, j], solution.MWind[:, j], solution.RMWind[:, j],
                          solution.MDischarge[:, j], solution.MDeficit[:, j], -1 * solution.MSpillage[:, j], Topology[j], -1 * solution.MCharge[:, j],
                          solution.MStorage[:, j],(solution.MWind[:,j]-solution.RMWind[:, j]), solution.RMDeficit[:, j], 
                          solution.RMStorage[:, j], solution.RMDischarge[:, j],-1*solution.RMCharge[:, j],-1*solution.RMSpillage[:, j]])
            C = np.around(C.transpose())

            C = np.insert(C.astype('str'), 0, datentime, axis=1)
            if RSim is None: np.savetxt('Results/S{}{}-{}-{}.csv'.format(scenario, solution.Nodel[j], stormZone, n_year), C, fmt='%s', delimiter=',', header=header, comments='')
            else: pass
                # C = C[max(0, RSim[1] - solution.stormDur.max() - 672):RSim[1] + 49, :] #2 weeks before storm + 1 day after 
                # np.savetxt('Results/S-Deficit{}{}-{}-{}-{}.csv'.format(scenario, solution.Nodel[j], stormZone, n_year, RSim[0]), C, fmt='%s', delimiter=',', header=header, comments='')

    print('Load profiles and generation mix is produced.')

    return True

def GGTA(solution):
    """GW, GWh, TWh p.a. and A$/MWh information"""

    factor = np.genfromtxt('Data/factor.csv', dtype=None, delimiter=',', encoding=None)
    factor = dict(factor)

    CPV, CWind, CPHP, CPHS = (sum(solution.CPV), sum(solution.CWind), sum(solution.CPHP), solution.CPHS) # GW, GWh
    CapHydro, CapBio = CHydro.sum(), CBio.sum() # GW
    CapHydrobio = CapHydro + CapBio

    GPV, GWind, GHydro, GBio = map(lambda x: x * pow(10, -6) * resolution / years, (solution.GPV.sum(), solution.GWind.sum(), solution.MHydro.sum(), solution.MBio.sum())) # TWh p.a.
    GHydrobio = GHydro + GBio
    CFPV, CFWind = (GPV / CPV / 8.76, GWind / CWind / 8.76)

    CostPV = factor['PV'] * CPV # A$b p.a.
    CostWind = factor['Wind'] * CWind # A$b p.a.
    CostHydro = factor['Hydro'] * GHydro # A$b p.a.
    CostBio = factor['Hydro'] * GBio # A$b p.a.
    CostPH = factor['PHP'] * CPHP + factor['PHS'] * CPHS # A$b p.a.
    if scenario>=21:
        CostPH -= factor['LegPH']

    CostDC = np.array([factor['FQ'], factor['NQ'], factor['NS'], factor['NV'], factor['AS'], factor['SW'], factor['TV']])
    CostDC = (CostDC * solution.CDC).sum() # A$b p.a.
    if scenario>=21:
        CostDC -= factor['LegINTC']

    CostAC = factor['ACPV'] * CPV + factor['ACWind'] * CWind # A$b p.a.

    Energy = (MLoad + MLoadD).sum() * pow(10, -9) * resolution / years # PWh p.a.
    Loss = np.sum(abs(solution.TDC), axis=0) * DCloss
    Loss = Loss.sum() * pow(10, -9) * resolution / years # PWh p.a.

    LCOE = (CostPV + CostWind + CostHydro + CostBio + CostPH + CostDC + CostAC) / (Energy - Loss)
    LCOG = (CostPV + CostWind + CostHydro + CostBio) * pow(10, 3) / (GPV + GWind + GHydro + GBio)
    LCOGP = CostPV * pow(10, 3) / GPV if GPV!=0 else 0
    LCOGW = CostWind * pow(10, 3) / GWind if GWind!=0 else 0
    LCOGH = CostHydro * pow(10, 3) / GHydro if GHydro!=0 else 0
    LCOGB = CostBio * pow(10, 3) / GBio if GBio!=0 else 0

    LCOB = LCOE - LCOG
    LCOBS = CostPH / (Energy - Loss)
    LCOBT = (CostDC + CostAC) / (Energy - Loss)
    LCOBL = LCOB - LCOBS - LCOBT

    print('Levelised costs of electricity:')
    print('\u2022 LCOE:', LCOE)
    print('\u2022 LCOG:', LCOG)
    print('\u2022 LCOB:', LCOB)
    print('\u2022 LCOG-PV:', LCOGP, '(%s)' % CFPV)
    print('\u2022 LCOG-Wind:', LCOGW, '(%s)' % CFWind)
    print('\u2022 LCOG-Hydro:', LCOGH)
    print('\u2022 LCOG-Bio:', LCOGB)
    print('\u2022 LCOB-Storage:', LCOBS)
    print('\u2022 LCOB-Transmission:', LCOBT)
    print('\u2022 LCOB-Spillage & loss:', LCOBL)

    D = np.zeros((1, 22))
    D[0, :] = ([Energy * pow(10, 3), Loss * pow(10, 3), CPV, GPV, CWind, GWind, CapHydrobio, GHydrobio, CPHP, CPHS] 
              + list(solution.CDC) 
              + [LCOE, LCOG, LCOBS, LCOBT, LCOBL])

    np.savetxt('Results/GGTA{}-{}-{}.csv'.format(scenario,stormZone, n_year), D, fmt='%f', delimiter=',')
    print('Energy generation, storage and transmission information is produced.')

    return True

def TransmissionFactors(solution, flexible, resilience = False):
    if resilience == True: raise NotImplementedError("Even if you're doing FIRM_resilience, don't use this")
    
    if scenario>=21:
        if resilience: 
            solution.TDC, solution.TDCR = Transmission(solution, output=True, resilience=True) # TDC(t, k), MW
        else: 
            solution.TDC = Transmission(solution, output=True, resilience=False) # TDC(t, k), MW
    else:
        solution.TDC = np.zeros((intervals, len(DCloss))) # TDC(t, k), MW
        
        solution.MPeak = np.tile(flexible, (nodes, 1)).transpose() # MW
        solution.MBaseload = GBaseload.copy() # MW

        solution.MPV = solution.GPV.sum(axis=1) if solution.GPV.shape[1]>0 else np.zeros((intervals, 1))
        solution.MWind = solution.GWind.sum(axis=1) if solution.GWind.shape[1]>0 else np.zeros((intervals, 1))

        solution.MDischarge = np.tile(solution.Discharge, (nodes, 1)).transpose()
        solution.MDeficit = np.tile(solution.Deficit, (nodes, 1)).transpose()
        solution.MCharge = np.tile(solution.Charge, (nodes, 1)).transpose()
        solution.MStorage = np.tile(solution.Storage, (nodes, 1)).transpose()
        solution.MSpillage = np.tile(solution.Spillage, (nodes, 1)).transpose()

        solution.MChargeD = np.tile(solution.ChargeD, (nodes, 1)).transpose()
        solution.MP2V = np.tile(solution.P2V, (nodes, 1)).transpose()
        
        if resilience: 
            solution.TDCR = np.zeros((intervals, len(DCloss))) # TDC(t, k), MW
            solution.MWindR = solution.GWindR.sum(axis=1) if solution.GWindR.shape[1]>0 else np.zeros((intervals, 1))
            
            solution.MDischargeR = np.tile(solution.RDischarge, (nodes, 1)).transpose()
            solution.MDeficitR = np.tile(solution.RDeficit, (nodes, 1)).transpose()
            solution.MChargeR = np.tile(solution.RCharge, (nodes, 1)).transpose()
            solution.MStorageR = np.tile(solution.RStorage, (nodes, 1)).transpose()
            solution.MSpillageR = np.tile(solution.RSpillage, (nodes, 1)).transpose()

            solution.MChargeDR = np.tile(solution.RChargeD, (nodes, 1)).transpose()
            solution.MP2VR = np.tile(solution.RP2V, (nodes, 1)).transpose()
            
            
    solution.CDC = np.amax(abs(solution.TDC), axis = 0) * pow(10, -3)
    solution.FQ, solution.NQ, solution.NS, solution.NV, solution.AS, solution.SW, solution.TV = map(lambda k: solution.TDC[:, k], range(solution.TDC.shape[1]))

    solution.MHydro = np.tile(CHydro - CBaseload, (intervals, 1)) * pow(10, 3) # GW to MW
    solution.MHydro = np.minimum(solution.MHydro, solution.MPeak)
    solution.MBio = solution.MPeak - solution.MHydro
    solution.MHydro += solution.MBaseload

    solution.Topology = np.array(
        [-1 *solution.FQ, -1 *(solution.NQ +solution.NS +solution.NV), -1 *solution.AS ,solution.FQ +solution.NQ, 
         solution.NS + solution.AS -solution.SW, -1 *solution.TV,solution.NV +solution.TV,solution.SW])
    return solution
    
def DeficitAnalysis(capacities, flexible, N=1):   
    if N is None: return True
    solution = Solution(capacities)
    Deficit, DeficitD, RDeficit, RDeficitD = Resilience(solution, flexible=flexible)
    topNDeficitIndx = np.flip(np.argpartition(RDeficit, -N)[-N:])
    
    for n, indx in enumerate(topNDeficitIndx):
        solution = Solution(capacities)
        solution = Resilience(solution, flexible, RSim=indx, output='solution')
        solution = TransmissionFactors(solution, flexible)
        LPGM(solution, (n, indx))
    return True 

def Information(x, flexible, NDeficitAnalysis=None, resilience=False):
    """Dispatch: Statistics.Information(x, Flex)"""

    start = dt.datetime.now()
    print("Statistics start at", start)

    # assert verifyDispatch(x, flexible, resilience)

    S = Solution(x)
    Deficit, DeficitD, RDeficit, RDeficitD = Resilience(S, flexible=flexible)

    S = TransmissionFactors(S, flexible)
    if not resilience: Debug(S)
    
    if not resilience: GGTA(S) 

    end = dt.datetime.now()
    print("Statistics took", end - start)

    return True

def DeficitInformation(capacities, flexible, NDeficitAnalysis=None):
    start =  dt.datetime.now()
    print("Deficit Statistics start at", start)
    
    S = Solution(capacities)
    Deficit, DeficitD, RDeficit, RDeficitD = Resilience(S, flexible=flexible)

    S = TransmissionFactors(S, flexible)
    LPGM(S)
    DeficitAnalysis(capacities, flexible, NDeficitAnalysis)

    end = dt.datetime.now()
    print("Deficit Statistics took", end - start)
    return True
   
    
    
def verifyDispatch(capacities, flexible, resilience=False):
    S = Solution(capacities)
    
    Deficit, DeficitD, RDeficit, RDeficitD = Resilience(S, flexible=flexible)
    if resilience: assert (RDeficit + RDeficitD).sum() * resolution < 0.1, f'R - Energy generation and demand are not balanced. deficit = {round((RDeficit + RDeficitD).sum() * resolution,2)}'
    else: assert (Deficit + DeficitD).sum() * resolution < 0.1, f'Energy generation and demand are not balanced. deficit = {round((Deficit + DeficitD).sum() * resolution,2)}'
    
    return True



if __name__ == '__main__': 
    capacities = np.genfromtxt('Results/Optimisation_resultx{}-{}-{}.csv'.format(scenario, stormZone, n_year), delimiter=',')
    flexible = np.genfromtxt('Results/Dispatch_Flexible{}-{}-{}.csv'.format(scenario, stormZone, n_year), delimiter=',', skip_header=1)
    Information(capacities, flexible, 5)
