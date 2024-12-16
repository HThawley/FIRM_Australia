# Load profiles and generation mix data (LPGM) & energy generation, storage and transmission information (GGTA)
# based on x/capacities from Optimisation and flexible from Dispatch
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

from Input import *
from Simulation import Reliability
from Network import Transmission
from Costs import pv_costs, onsw_costs, offsw_costs, ACgen_costs, phes_costs, hydro_purchase, hvdc_costs


import numpy as np
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta as td

def Debug(solution):
    """Debugging"""

    Load, PV, OnsW, OffsW = solution.MLoad.sum(axis=1), solution.GPV.sum(axis=1), solution.GOnsW.sum(axis=1), solution.GOffsW.sum(axis=1)
    Baseload, Peak = solution.MBaseload.sum(axis=1), solution.MPeak.sum(axis=1)

    Discharge, Charge, Storage = solution.Discharge, solution.Charge, solution.Storage
    Deficit, Spillage = solution.Deficit, solution.Spillage

    PHS = solution.CPHS * pow(10, 3)  # GWh to MWh
    efficiency = solution.efficiency

    for i in range(intervals):
        # Energy supply-demand balance
        assert abs(Load[i] + Charge[i] + Spillage[i] - PV[i] - OnsW[i] - OffsW[i] - Baseload[i] 
                   - Peak[i] - Discharge[i] - Deficit[i]) <= 1

        # Discharge, Charge and Storage
        if i==0:
            assert abs(Storage[i] - 0.5 * PHS + Discharge[i] * resolution - Charge[i] * resolution * efficiency) <= 1
        else:
            assert abs(Storage[i] - Storage[i - 1] + Discharge[i] * resolution - Charge[i] * resolution * efficiency) <= 1

    # Capacity: PV, wind, Discharge, Charge and Storage
    assert np.amax(PV)    <= 1.005*sum(solution.CPV)    * pow(10, 3), print(np.amax(PV)    - sum(solution.CPV)    * pow(10, 3))
    assert np.amax(OnsW)  <= 1.005*sum(solution.COnsW)  * pow(10, 3), print(np.amax(OnsW)  - sum(solution.COnsW)  * pow(10, 3))
    assert np.amax(OffsW) <= 1.005*sum(solution.COffsW) * pow(10, 3), print(np.amax(OffsW) - sum(solution.COffsW) * pow(10, 3))

    assert np.amax(Discharge) <= 1.005*sum(solution.CPHP) * pow(10, 3), print(np.amax(Discharge) - sum(solution.CPHP) * pow(10, 3))
    assert np.amax(Charge)    <= 1.005*sum(solution.CPHP) * pow(10, 3), print(np.amax(Charge)    - sum(solution.CPHP) * pow(10, 3))
    assert np.amax(Storage)   <= 1.005*solution.CPHS      * pow(10, 3), print(np.amax(Storage)   - solution.CPHS      * pow(10, 3))

    print('Debugging: everything is ok')

    return True

def LPGM(solution):
    """Load profiles and generation mix data"""

    C = np.stack([solution.MLoad.sum(axis=1), solution.MHydro.sum(axis=1), solution.MBio.sum(axis=1), 
                  solution.GPV.sum(axis=1), solution.GOnsW.sum(axis=1), solution.GOffsW.sum(axis=1),
                  solution.Discharge, solution.Deficit, -solution.Spillage, -solution.Charge,
                  solution.Storage,
                  solution.FQ, solution.NQ, solution.NS, solution.NV, solution.AS, solution.SW, solution.TV])
    C = np.around(C.T)

    datentime = pd.date_range(f'{firstyear}/01/01 00:00:00', f'{finalyear}/12/31 23:30:01', freq='30min')
    datentime = datentime[~((datentime.month == 2) & (datentime.day == 29))] # remove leap days
    datentime = datentime.strftime('%a %d-%b %Y %H:%M')
    C = np.insert(C.astype('str'), 0, datentime, axis=1)

    header = ','.join(['Date & time','Operational demand','Hydropower','Biomass',
                       'Solar photovoltaics','Onshore Wind','Offshore Wind','Pumped hydro',
                       'Energy deficit','Energy spillage','PHES-Charge','PHES-Storage',
                       'FNQ-QLD','NSW-QLD','NSW-SA','NSW-VIC','NT-SA','SA-WA','TAS-VIC'])
    
    np.savetxt('Results/S{}.csv'.format(scenario), C, fmt='%s', delimiter=',', header=header, comments='')

    if scenario>=21:
        header = ','.join(['Date & time','Operational demand','Hydropower','Biomass',
                           'Solar photovoltaics','Onshore Wind','Offshore Wind','Pumped hydro',
                           'Energy deficit','Energy spillage','Transmission','PHES-Charge','PHES-Storage'])
        
        Topology = solution.Topology[np.where(np.in1d(np.array(['FNQ', 'NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC', 'WA']), coverage) == True)[0]]

        for j in range(nodes):
            C = np.stack([solution.MLoad[:, j], solution.MLoad[:, j], solution.MHydro[:, j], 
                          solution.MBio[:, j], solution.MPV[:, j], solution.MOnsW[:, j], solution.MOffsW[:, j],
                          solution.MDischarge[:, j], solution.MDeficit[:, j], -solution.MSpillage[:, j], 
                          Topology[j], -solution.MCharge[:, j], solution.MStorage[:, j]])
            C = np.around(C.T)

            C = np.insert(C.astype('str'), 0, datentime, axis=1)
            np.savetxt('Results/S{}{}.csv'.format(scenario, solution.Nodel[j]), C, fmt='%s', delimiter=',', header=header, comments='')

    print('Load profiles and generation mix is produced.')

    return True

def GGTA(solution):
    """GW, GWh, TWh p.a. and A$/MWh information"""

    CPV, COnsW, COffsW, CPHP, CPHS = (sum(solution.CPV), sum(solution.COnsW), sum(solution.COffsW), sum(solution.CPHP), solution.CPHS) # GW, GWh
    CapHydro, CapBio = CHydro.sum(), CBio.sum() # GW
    CapHydrobio = CapHydro + CapBio

    GPV, GOnsW, GOffsW, GHydro, GBio, GPHES = map(lambda x: x * pow(10, -6) * resolution / years, 
                                           (solution.GPV.sum(), solution.GOnsW.sum(), solution.GOffsW.sum(), 
                                            solution.MHydro.sum(), solution.MBio.sum(), solution.MDischarge.sum())) # TWh p.a.
    GHydrobio = GHydro + GBio
    CFPV, CFOnsW, CFOffsW = (GPV / CPV / 8.76, GOnsW / COnsW / 8.76, GOffsW / COffsW / 8.76)

    CostPV    = pv_costs    * CPV    * pow(10, -9) # A$b p.a.
    CostOnsW  = onsw_costs  * COnsW  * pow(10, -9) # A$b p.a.
    CostOffsW = offsw_costs * COffsW * pow(10, -9) # A$b p.a.
    CostHydro = hydro_purchase * GHydro * pow(10, -9) # A$b p.a.
    CostBio   = hydro_purchase * GBio   * pow(10, -9)  # A$b p.a.
    CostPH    = (phes_costs[0] * CPHP 
                 + phes_costs[1] * CPHS 
                 + phes_costs[2]) * pow(10, -9) # A$b p.a.

    CostDC = (hvdc_costs * solution.CDC).sum() * pow(10, -9) # A$b p.a.

    CostAC = ACgen_costs * (CPV + COnsW + COffsW) * pow(10, -9) # A$b p.a.

    Energy = MLoad.sum() * pow(10, -9) * resolution / years # PWh p.a.
    Loss = np.sum(abs(solution.TDC), axis=0) * DCloss
    Loss = Loss.sum() * pow(10, -9) * resolution / years # PWh p.a.

    LCOE = (CostPV + CostOnsW + CostOffsW + CostHydro + CostBio + CostPH + CostDC + CostAC) / (Energy - Loss)
    LCOG = (CostPV +  CostOnsW + CostOffsW + CostHydro + CostBio) * pow(10, 3) / (GPV + GOnsW + GOffsW + GHydro + GBio)
    LCOGP     = CostPV    * pow(10, 3) / GPV    if GPV!=0    else 0
    LCOGOnsW  = CostOnsW  * pow(10, 3) / GOnsW  if GOnsW!=0  else 0
    LCOGOffsW = CostOffsW * pow(10, 3) / GOffsW if GOffsW!=0 else 0
    LCOGH     = CostHydro * pow(10, 3) / GHydro if GHydro!=0 else 0
    LCOGB     = CostBio   * pow(10, 3) / GBio   if GBio!=0   else 0

    LCOB = LCOE - LCOG
    LCOBS = CostPH / (Energy - Loss)
    LCOBT = (CostDC + CostAC) / (Energy - Loss)
    LCOBL = LCOB - LCOBS - LCOBT

    print('Levelised costs of electricity:')
    print('\u2022 LCOE:', LCOE)
    print('\u2022 LCOG:', LCOG)
    print('\u2022 LCOB:', LCOB)
    print('\u2022 LCOG-PV:', LCOGP, '(%s)' % CFPV)
    print('\u2022 LCOG-Onshore Wind:', LCOGOnsW, '(%s)' % CFOnsW)
    print('\u2022 LCOG-Offshore Wind:', LCOGOffsW, '(%s)' % CFOffsW)
    print('\u2022 LCOG-Hydro:', LCOGH)
    print('\u2022 LCOG-Bio:', LCOGB)
    print('\u2022 LCOB-Storage:', LCOBS)
    print('\u2022 LCOB-Transmission:', LCOBT)
    print('\u2022 LCOB-Spillage & loss:', LCOBL)

    D = np.array([Energy * pow(10, 3), Loss * pow(10, 3), CPV, GPV, COnsW, GOnsW, COffsW, GOffsW, CapHydrobio, GHydrobio, CPHP, CPHS, GPHES] \
              + list(solution.CDC) \
              + [LCOE, LCOG, LCOBS, LCOBT, LCOBL])

    header = ','.join(['Demand Served', 'Transmission Loss', 'Utility PV (GW)', 'Utility PV (TWh p.a.)', 
                       'Onshore Wind (GW)', 'Onshore Wind (TWh p.a.)', 'Offshore Wind (GW)', 
                       'Offshore Wind (TWh p.a.)', 'Hydro&Bio (GW)', 'Hydro&Bio (TWh p.a.)', 
                       'Pumped Hydro (GW)', 'Pumped Hydro (GWh)', 'Pumped Hydro (TWh p.a.)',
                       'FNQ-QLD (GW)','NSW-QLD (GW)','NSW-SA (GW)','NSW-VIC (GW)','NT-SA (GW)',
                       'SA-WA (GW)','TAS-VIC (GW)','LCOE', 'LCOG', 'LCOB - storage', 
                       'LCOB - Transmission&Distribution', 'LCOB - Curtailments and other losses'])


    np.savetxt('Results/GGTA{}.csv'.format(scenario), D.reshape(1,-1), fmt='%s', delimiter=',', header=header, comments='')
    print('Energy generation, storage and transmission information is produced.')

    return True

def Information(x, flexible):
    """Dispatch: Statistics.Information(x, Flex)"""

    start = dt.now()
    print("Statistics start at", start)

    S = Solution(x)
    Deficit = Reliability(S, flexible=flexible)

    try:
        assert Deficit.sum() * resolution < 0.1, 'Energy generation and demand are not balanced.'
    except AssertionError:
        pass

    if scenario>=21:
        S.TDC = Transmission(S, output=True) # TDC(t, k), MW
    else:
        S.TDC = np.zeros((intervals, len(DCloss))) # TDC(t, k), MW

        S.MPeak = np.tile(flexible, (nodes, 1)).transpose() # MW
        S.MBaseload = GBaseload.copy() # MW

        S.MPV    = S.GPV.sum(axis=1)    if S.GPV.shape[1]>0    else np.zeros((intervals, 1))
        S.MOnsW  = S.GOnsW.sum(axis=1)  if S.GOnsW.shape[1]>0  else np.zeros((intervals, 1))
        S.MOffsW = S.GOffsW.sum(axis=1) if S.GOffsW.shape[1]>0 else np.zeros((intervals, 1))

        S.MDischarge = np.tile(S.Discharge, (nodes, 1)).T
        S.MDeficit   = np.tile(S.Deficit,   (nodes, 1)).T
        S.MCharge    = np.tile(S.Charge,    (nodes, 1)).T
        S.MStorage   = np.tile(S.Storage,   (nodes, 1)).T
        S.MSpillage  = np.tile(S.Spillage,  (nodes, 1)).T

    S.CDC = np.amax(abs(S.TDC), axis=0) * pow(10, -3) # CDC(k), MW to GW
    S.FQ, S.NQ, S.NS, S.NV, S.AS, S.SW, S.TV = map(lambda k: S.TDC[:, k], range(S.TDC.shape[1]))

    S.MHydro = np.tile(CHydro - CBaseload, (intervals, 1)) * pow(10, 3) # GW to MW
    S.MHydro = np.minimum(S.MHydro, S.MPeak)
    S.MBio = S.MPeak - S.MHydro
    S.MHydro += S.MBaseload

    S.Topology = np.array([-1 * S.FQ, -1 * (S.NQ + S.NS + S.NV), -1 * S.AS, S.FQ + S.NQ, S.NS + S.AS - S.SW, -1 * S.TV, S.NV + S.TV, S.SW])

    Debug(S)
    LPGM(S)
    GGTA(S)

    end = dt.now()
    print("Statistics took", end - start)

    return True

if __name__ == '__main__':
    capacities = np.genfromtxt('Results/Optimisation_resultx{}.csv'.format(scenario), delimiter=',')
    flexible = np.genfromtxt('Results/Dispatch_Flexible{}.csv'.format(scenario), delimiter=',', skip_header=1)
    Information(capacities, flexible)