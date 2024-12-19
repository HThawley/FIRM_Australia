# Load profiles and generation mix data (LPGM) & energy generation, storage and transmission information (GGTA)
# based on x/capacities from Optimisation and flexible from Dispatch
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

from Costs import *
from Input import *
from Simulation import Reliability
from Network import Transmission

import numpy as np
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta as td
from warnings import warn

def Debug(solution):
    """Debugging"""

    Load, PV, OnsW, OffW = solution.MOLoad.sum(axis=1), solution.GPV.sum(axis=1), solution.GOnsW.sum(axis=1), solution.GOffW.sum(axis=1)
    Baseload, Peak = solution.GBaseload.sum(axis=1), solution.MPeak.sum(axis=1)

    Discharge, Charge, Storage = solution.Discharge, solution.Charge, solution.Storage
    Deficit, Spillage = solution.Deficit, solution.Spillage
    if unbalanced:=solution.Deficit.sum() > 0:
        warn(f"Supply/demand is not balanced. Deficit: {solution.Deficit.sum()*resolution/years} MWh p.a.", UserWarning)
    PHS = solution.CPHS * 1000  # GWh to MWh
    efficiency = solution.efficiency

    for i in range(intervals):
        # Energy supply-demand balance
        assert abs(Load[i] + Charge[i] + Spillage[i] - PV[i] - OnsW[i] - OffW[i] - Baseload[i] 
                   - Peak[i] - Discharge[i] - Deficit[i]) <= 1

        # Discharge, Charge and Storage
        if i==0:
            assert abs(Storage[i] - 0.5 * PHS + Discharge[i] * resolution - Charge[i] * resolution * efficiency) <= 1
        else:
            assert abs(Storage[i] - Storage[i - 1] + Discharge[i] * resolution - Charge[i] * resolution * efficiency) <= 1

    # Capacity: PV, wind, Discharge, Charge and Storage
    assert np.amax(PV)   <= 1.005*sum(solution.CPV)   * 1000, print(np.amax(PV)   - sum(solution.CPV)   * 1000)
    assert np.amax(OnsW) <= 1.005*sum(solution.COnsW) * 1000, print(np.amax(OnsW) - sum(solution.COnsW) * 1000)
    assert np.amax(OffW) <= 1.005*sum(solution.COffW) * 1000, print(np.amax(OffW) - sum(solution.COffW) * 1000)

    assert np.amax(Discharge) <= 1.005*sum(solution.CPHP) * 1000, print(np.amax(Discharge) - sum(solution.CPHP) * 1000)
    assert np.amax(Charge)    <= 1.005*sum(solution.CPHP) * 1000, print(np.amax(Charge)    - sum(solution.CPHP) * 1000)
    assert np.amax(Storage)   <= 1.005*solution.CPHS      * 1000, print(np.amax(Storage)   - solution.CPHS      * 1000)

    print('Debugging: operations behave as expected')

    return True

def LPGM(solution):
    """Load profiles and generation mix data"""

    C = np.stack([MLoad.sum(axis=1), MPVnsg.sum(axis=1), solution.MOLoad.sum(axis=1), 
                  solution.MHydro.sum(axis=1), solution.MBio.sum(axis=1), solution.GPV.sum(axis=1),
                  solution.GOnsW.sum(axis=1), solution.GOffW.sum(axis=1), solution.Discharge, 
                  solution.Deficit, -solution.Spillage, -solution.Charge, solution.Storage,
                  solution.FQ, solution.NQ, solution.NS, solution.NV, solution.AS, solution.SW, solution.TV])
    C = np.around(C.T)

    datentime = pd.date_range(f'{firstyear}/01/01 00:00:00', f'{finalyear}/12/31 23:30:01', freq='30min')
    datentime = datentime[~((datentime.month == 2) & (datentime.day == 29))] # remove leap days
    datentime = datentime.strftime('%a %d-%b %Y %H:%M')
    C = np.insert(C.astype('str'), 0, datentime, axis=1)

    header = ','.join(['Date & time','All demand','Rooftop Solar','Operational Demand','Hydropower',
                       'Biomass','Utility Solar','Onshore Wind','Offshore Wind','Pumped hydro',
                       'Energy deficit','Energy spillage','PHES-Charge','PHES-Storage',
                       'FNQ-QLD','NSW-QLD','NSW-SA','NSW-VIC','NT-SA','SA-WA','TAS-VIC'])
    
    np.savetxt(f'Results/S{suffix}.csv', C, fmt='%s', delimiter=',', header=header, comments='')

    if scenario>=21:
        header = ','.join(['Date & time','All demand','Rooftop Solar','Operational demand','Hydropower','Biomass',
                           'Solar photovoltaics','Onshore Wind','Offshore Wind','Pumped hydro',
                           'Energy deficit','Energy spillage','Transmission','PHES-Charge','PHES-Storage'])
        
        for j in range(nodes):
            C = np.stack([MLoad[:, j], MPVnsg[:, j], solution.MOLoad[:, j], solution.MHydro[:, j], 
                          solution.MBio[:, j], solution.MPV[:, j], solution.MOnsW[:, j], solution.MOffW[:, j],
                          solution.MDischarge[:, j], solution.MDeficit[:, j], -solution.MSpillage[:, j], 
                          solution.MImport[j], -solution.MCharge[:, j], solution.MStorage[:, j]])
            C = np.around(C.T)

            C = np.insert(C.astype('str'), 0, datentime, axis=1)
            np.savetxt(f'Results/S{Nodel[j]}{suffix}.csv', C, fmt='%s', delimiter=',', header=header, comments='')

    print('Load profiles and generation mix is produced.')

    return True

def GGTA(solution):
    """GW, GWh, TWh p.a. and A$/MWh information"""

    CPV, COnsW, COffW, CPHP, CPHS = (sum(solution.CPV), sum(solution.COnsW), sum(solution.COffW), sum(solution.CPHP), solution.CPHS) # GW, GWh
    CapHydro, CapBio = CHydro.sum(), CBio.sum() # GW
    CapHydrobio = CapHydro + CapBio

    GPV, GOnsW, GOffW, GHydro, GBio, GPHES = map(lambda x: x * 0.000_001 * resolution / years, 
                                           (solution.GPV.sum(), solution.GOnsW.sum(), solution.GOffW.sum(), 
                                            solution.MHydro.sum(), solution.MBio.sum(), solution.MDischarge.sum())) # TWh p.a.
    GHydrobio = GHydro + GBio
    CFPV, CFOnsW, CFOffW = (G/C/0.0876 for G, C in zip((GPV, GOnsW, GOffW), (CPV, COnsW, COffW)))

    CostPV    = costs.pv   * CPV   * pow(10, -9) # A$b p.a.
    CostOnsW  = costs.onsw * COnsW * pow(10, -9) # A$b p.a.
    CostOffW  = costs.offw * COffW * pow(10, -9) # A$b p.a.
    CostHydro = costs.hydro * GHydro * 0.001 # A$b p.a.
    CostBio   = costs.hydro * GBio   * 0.001 # A$b p.a.
    CostPH    = (costs.phes[0] * CPHP 
                 + costs.phes[1] * CPHS 
                 # + costs.phes[2] * GPHES * pow(10, 6)
                 + costs.phes[3]) * pow(10, -9) # A$b p.a.

    CostDC = (costs.hvdc * solution.CDC).sum() * pow(10, -9) # A$b p.a.

    CostAC = costs.ac * (CPV + COnsW + COffW) * pow(10, -9) # A$b p.a.

    Energy = solution.MOLoad.sum() * pow(10, -9) * resolution / years # PWh p.a.
    Loss = np.sum(abs(solution.TDC), axis=0) * DCloss
    Loss = Loss.sum() * pow(10, -9) * resolution / years # PWh p.a.

    LCOE = (CostPV + CostOnsW + CostOffW + CostHydro + CostBio + CostPH + CostDC + CostAC) / (Energy - Loss)
    LCOG = (CostPV +  CostOnsW + CostOffW + CostHydro + CostBio) * 1000 / (GPV + GOnsW + GOffW + GHydro + GBio)
    LCOGP    = CostPV    * 1000 / GPV    if GPV!=0    else 0
    LCOGOnsW = CostOnsW  * 1000 / GOnsW  if GOnsW!=0  else 0
    LCOGOffW = CostOffW  * 1000 / GOffW  if GOffW!=0  else 0
    LCOGH    = CostHydro * 1000 / GHydro if GHydro!=0 else 0
    LCOGB    = CostBio   * 1000 / GBio   if GBio!=0   else 0

    LCOB = LCOE - LCOG
    LCOBS = CostPH / (Energy - Loss)
    LCOBT = (CostDC + CostAC) / (Energy - Loss)
    LCOBL = LCOB - LCOBS - LCOBT

    print('Levelised costs of electricity:')
    print(f'\u2022 LCOE: {LCOE}')
    print(f'\u2022 LCOG: {LCOG}')
    print(f'\u2022 LCOB: {LCOB}')
    print(f'\u2022 LCOG-PV: {LCOGP}, (CF:{round(CFPV,3)}%)')
    print(f'\u2022 LCOG-Onshore Wind: {LCOGOnsW} (CF:{round(CFOnsW,3)}%)')
    print(f'\u2022 LCOG-Offshore Wind: {LCOGOffW} (CF:{round(CFOffW,3)}%)')
    print(f'\u2022 LCOG-Hydro: {LCOGH}')
    print(f'\u2022 LCOG-Bio: {LCOGB}')
    print(f'\u2022 LCOB-Storage: {LCOBS}')
    print(f'\u2022 LCOB-Transmission: {LCOBT}')
    print(f'\u2022 LCOB-Spillage & loss: {LCOBL}')

    D = np.array([Energy * 1000 + GPVnsg * 0.000_001, Loss * 1000, 
                  GPVnsg * 0.000_001, CPV, GPV, COnsW, GOnsW, COffW,  
                  GOffW, CapHydrobio, GHydrobio, CPHP, CPHS, GPHES]
              + list(solution.CDC)
              + [LCOE, LCOG, LCOBS, LCOBT, LCOBL])

    header = ','.join(['Demand Served (TWh p.a.)', 'Transmission Loss (TWh p.a.)', 
                       'Rooftop PV (TWh p.a)','Utility PV (GW)', 'Utility PV (TWh p.a.)', 
                       'Onshore Wind (GW)', 'Onshore Wind (TWh p.a.)', 'Offshore Wind (GW)', 
                       'Offshore Wind (TWh p.a.)', 'Hydro&Bio (GW)', 'Hydro&Bio (TWh p.a.)', 
                       'Pumped Hydro (GW)', 'Pumped Hydro (GWh)', 'Pumped Hydro (TWh p.a.)',
                       'FNQ-QLD (GW)','NSW-QLD (GW)','NSW-SA (GW)','NSW-VIC (GW)','NT-SA (GW)',
                       'SA-WA (GW)','TAS-VIC (GW)','LCOE', 'LCOG', 'LCOB - storage', 
                       'LCOB - Transmission&Distribution', 'LCOB - Curtailments and other losses'])


    np.savetxt(f'Results/GGTA{suffix}.csv', D.reshape(1,-1), fmt='%s', delimiter=',', header=header, comments='')
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
        S.TDC = Transmission(S) # TDC(t, k), MW
    else:
        S.TDC = np.zeros((intervals, len(DCloss))) # TDC(t, k), MW
        S.MImport = np.zeros((intervals, nodes))
        
        S.MPeak = np.tile(flexible, (nodes, 1)).transpose() # MW

        S.MPV   = S.GPV.sum(axis=1)   if S.GPV.shape[1]>0   else np.zeros((intervals, 1))
        S.MOnsW = S.GOnsW.sum(axis=1) if S.GOnsW.shape[1]>0 else np.zeros((intervals, 1))
        S.MOffW = S.GOffW.sum(axis=1) if S.GOffW.shape[1]>0 else np.zeros((intervals, 1))

        S.MDischarge = S.Discharge.reshape(-1,1)
        S.MDeficit   = S.Deficit.reshape(-1,1)
        S.MCharge    = S.Charge.reshape(-1,1)
        S.MStorage   = S.Storage.reshape(-1,1)
        S.MSpillage  = S.Spillage.reshape(-1,1)

    S.CDC = np.amax(abs(S.TDC), axis=0) * 0.001 # CDC(k), MW to GW
    S.FQ, S.NQ, S.NS, S.NV, S.AS, S.SW, S.TV = map(lambda k: S.TDC[:, k], range(S.TDC.shape[1]))

    S.MHydro = np.tile(CHydro - CBaseload, (intervals, 1)) * 1000 # GW to MW
    S.MHydro = np.minimum(S.MHydro, S.MPeak)
    S.MBio = S.MPeak - S.MHydro
    S.MHydro += GBaseload


    Debug(S)
    LPGM(S)
    GGTA(S)

    end = dt.now()
    print("Statistics took", end - start)

    return True

if __name__ == '__main__':
    capacities = np.genfromtxt(f'Results/Optimisation_resultx{suffix}.csv', delimiter=',')
    flexible   = np.genfromtxt(f'Results/Dispatch_Flexible{suffix}.csv',    delimiter=',', skip_header=1)
    Information(capacities, flexible)