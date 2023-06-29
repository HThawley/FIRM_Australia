# Modelling input and assumptions
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

import numpy as np
import pandas as pd
from Optimisation import scenario
from Simulation import Reliability
from CoSimulation import Resilience
from Network import Transmission

Nodel = np.array(['FNQ', 'NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC', 'WA'])
PVl   = np.array(['NSW']*7 + ['FNQ']*1 + ['QLD']*2 + ['FNQ']*3 + ['SA']*6 + ['TAS']*0 + ['VIC']*1 + ['WA']*1 + ['NT']*1)
Windl = np.array(['NSW']*8 + ['FNQ']*1 + ['QLD']*2 + ['FNQ']*2 + ['SA']*8 + ['TAS']*4 + ['VIC']*4 + ['WA']*3 + ['NT']*1)
resolution = 0.5

MLoad = np.genfromtxt('Data/electricity.csv', delimiter=',', skip_header=1, usecols=range(4, 4+len(Nodel))) # EOLoad(t, j), MW
for i in ['evan', 'erigid', 'earticulated', 'enonfreight', 'ebus', 'emotorcycle', 'erail', 'eair', 'ewater', 'ecooking', 'emanufacturing', 'emining']:
    MLoad += np.genfromtxt('Data/{}.csv'.format(i), delimiter=',', skip_header=1, usecols=range(4, 4+len(Nodel)))

DSP = 0.8 if scenario>=31 else 0
MLoad += (1 - DSP) * np.genfromtxt('Data/ecar.csv', delimiter=',', skip_header=1, usecols=range(4, 4+len(Nodel)))

MLoadD = DSP * np.genfromtxt('Data/ecar.csv', delimiter=',', skip_header=1, usecols=range(4, 4+len(Nodel)))

TSPV = np.genfromtxt('Data/pv.csv', delimiter=',', skip_header=1, usecols=range(4, 4+len(PVl))) # TSPV(t, i), MW
TSWind = np.genfromtxt('Data/wind.csv', delimiter=',', skip_header=1, usecols=range(4, 4+len(Windl))) # TSWind(t, i), MW
windFrag = np.genfromtxt('Data/windFragility.csv', delimiter=',', skip_header=1, usecols=range(0,len(Windl)))
windFrag, stormDur = windFrag[0], windFrag[1].astype(int)

assets = np.genfromtxt('Data/hydrobio.csv', dtype=None, delimiter=',', encoding=None)[1:, 1:].astype(float)
CHydro, CBio = [assets[:, x] * pow(10, -3) for x in range(assets.shape[1])] # CHydro(j), MW to GW
CBaseload = np.array([0, 0, 0, 0, 0, 1.0, 0, 0]) # 24/7, GW
CPeak = CHydro + CBio - CBaseload # GW

cars = np.genfromtxt('Data/cars.csv', dtype=None, delimiter=',', encoding=None)[1:, 1:].astype(float)
CDP = DSP * cars[:, 0] * 9.6 * pow(10, -6) # kW to GW
CDS = DSP * cars[:, 0] * 77 * 0.75 * pow(10, -6) # kWh to GWh

# FQ, NQ, NS, NV, AS, SW, only TV constrained
CDC6max = 3 * 0.63 # GW
DCloss = np.array([1500, 1000, 1000, 800, 1200, 2400, 400]) * 0.03 * pow(10, -3)

efficiency = 0.8
efficiencyD = 0.8
factor = np.genfromtxt('Data/factor.csv', delimiter=',', usecols=1)

firstyear, finalyear, timestep = (2020, 2029, 1)

if scenario<=17:
    node = Nodel[scenario % 10]

    MLoad, MLoadD = [x[:, np.where(Nodel==node)[0]] for x in (MLoad, MLoadD)]
    TSPV = TSPV[:, np.where(PVl==node)[0]]
    TSWind = TSWind[:, np.where(Windl==node)[0]]
    windFrag = windFrag[np.where(Windl==node)[0]]
    windFrag = windFrag / windFrag.sum()
    stormDur = stormDur[np.where(Windl==node)[0]]
    
    CHydro, CBio, CBaseload, CPeak, CDP, CDS = [x[np.where(Nodel==node)[0]] for x in (CHydro, CBio, CBaseload, CPeak, CDP, CDS)]
    if node=='QLD':
        MLoad, MLoadD, CDP, CDS = [x / 0.9 for x in (MLoad, MLoadD, CDP, CDS)]

    Nodel, PVl, Windl = [x[np.where(x==node)[0]] for x in (Nodel, PVl, Windl)]

if scenario>=21:
    coverage = [np.array(['NSW', 'QLD', 'SA', 'TAS', 'VIC']),
                np.array(['NSW', 'QLD', 'SA', 'TAS', 'VIC', 'WA']),
                np.array(['NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC']),
                np.array(['NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC', 'WA']),
                np.array(['FNQ', 'NSW', 'QLD', 'SA', 'TAS', 'VIC']),
                np.array(['FNQ', 'NSW', 'QLD', 'SA', 'TAS', 'VIC', 'WA']),
                np.array(['FNQ', 'NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC']),
                np.array(['FNQ', 'NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC', 'WA'])][scenario % 10 - 1]

    MLoad, MLoadD = [x[:, np.where(np.in1d(Nodel, coverage)==True)[0]] for x in (MLoad, MLoadD)]
    TSPV = TSPV[:, np.where(np.in1d(PVl, coverage)==True)[0]]
    TSWind = TSWind[:, np.where(np.in1d(Windl, coverage)==True)[0]]
    windFrag = windFrag[np.where(np.in1d(Windl, coverage)==True)[0]]
    windFrag = windFrag / windFrag.sum()
    stormDur = stormDur[np.where(Windl==node)[0]]
    
    CHydro, CBio, CBaseload, CPeak, CDP, CDS = [x[np.where(np.in1d(Nodel, coverage)==True)[0]] for x in (CHydro, CBio, CBaseload, CPeak, CDP, CDS)]
    if 'FNQ' not in coverage:
        MLoad[:, np.where(coverage=='QLD')[0][0]] /= 0.9
        MLoadD[:, np.where(coverage=='QLD')[0][0]] /= 0.9
        CDP[np.where(coverage == 'QLD')[0]] /= 0.9
        CDS[np.where(coverage == 'QLD')[0]] /= 0.9

    Nodel, PVl, Windl = [x[np.where(np.in1d(x, coverage)==True)[0]] for x in (Nodel, PVl, Windl)]

intervals, nodes = MLoad.shape
years = int(resolution * intervals / 8760)
pzones, wzones = (TSPV.shape[1], TSWind.shape[1])
pidx, widx, sidx = (pzones, pzones + wzones, pzones + wzones + nodes)

energy = (MLoad + MLoadD).sum() * pow(10, -9) * resolution / years # PWh p.a.
contingency = list(0.25 * (MLoad + MLoadD).max(axis=0) * pow(10, -3)) # MW to GW

GBaseload = np.tile(CBaseload, (intervals, 1)) * pow(10, 3) # GW to MW

# Reorder windfrags, we want the most probable place to have the biggest reduction in output
windFrag = np.array(pd.Series(windFrag).map(dict(zip(np.sort(windFrag), np.flip(np.sort(windFrag))))))

try: 
    #if possible, use the original result as the first guess
    x0 = np.genfromtxt(f'OptimisationResultx/Optimisation_resultx{scenario}.csv', delimiter = ',')
except FileNotFoundError:
    x0 = None
    
def cost(solution): 

    Deficit, DeficitD = Reliability(solution, flexible=np.zeros(intervals)) # solutionj-EDE(t, j), MW
    Flexible = (Deficit + DeficitD / efficiencyD).sum() * resolution / years / (0.5 * (1 + efficiency)) # MWh p.a.
    Hydro = Flexible + GBaseload.sum() * resolution / years # Hydropower & biomass: MWh p.a.
    PenHydro = max(0, Hydro - 20 * pow(10, 6)) # TWh p.a. to MWh p.a.

    Deficit, DeficitD, RDeficit, RDeficitD = Resilience(solution, flexible=np.ones(intervals) * CPeak.sum() * pow(10, 3)) # solutionj-EDE(t, j), GW to MW
    PenDeficit = max(0, (Deficit + DeficitD / efficiencyD).sum() * resolution) # MWh
    StormDeficit = max(0, (RDeficit + RDeficitD / efficiencyD).sum() * resolution) #MWh

    TDC = Transmission(solution) if scenario>=21 else np.zeros((intervals, len(DCloss))) # TDC: TDC(t, k), MW
    CDC = np.amax(abs(TDC), axis=0) * pow(10, -3) # CDC(k), MW to GW
    PenDC = max(0, CDC[6] - CDC6max) * pow(10, 3) # GW to MW

    cost = factor * np.array([sum(solution.CPV), sum(solution.CWind), sum(solution.CPHP), solution.CPHS] + list(CDC) + [sum(solution.CPV), sum(solution.CWind), Hydro * pow(10, -6), -1, -1]) # $b p.a.
    if scenario<=17:
        cost[-1], cost[-2] = [0] * 2
    cost = cost.sum()
    loss = np.sum(abs(TDC), axis=0) * DCloss
    loss = loss.sum() * pow(10, -9) * resolution / years # PWh p.a.
    LCOE = cost / abs(energy - loss)

    penalties = PenHydro + PenDeficit + PenDC

    return StormDeficit, LCOE, penalties


#%%
class Solution:
    """A candidate solution of decision variables CPV(i), CWind(i), CPHP(j), S-CPHS(j)"""

    def __init__(self, x, stormZone):
        self.x = x
        self.stormZone = stormZone
        self.MLoad, self.MLoadD = (MLoad, MLoadD)
        self.intervals, self.nodes = (intervals, nodes)
        self.resolution = resolution

        self.CPV = list(x[: pidx]) # CPV(i), GW
        self.CWind = list(x[pidx: widx]) # CWind(i), GW        
        self.GPV = TSPV * np.tile(self.CPV, (intervals, 1)) * pow(10, 3) # GPV(i, t), GW to MW
        self.GWind = TSWind * np.tile(self.CWind, (intervals, 1)) * pow(10, 3) # GWind(i, t), GW to MW

        self.CPHP = list(x[widx: sidx]) # CPHP(j), GW
        self.CPHS = x[sidx] # S-CPHS(j), GWh
        self.CDP, self.CDS = (CDP, CDS)  # GW, GWh
        self.efficiency, self.efficiencyD = (efficiency, efficiencyD)

        self.Nodel, self.PVl, self.Windl = (Nodel, PVl, Windl)
        self.scenario = scenario

        self.GBaseload, self.CPeak = (GBaseload, CPeak)
        self.CHydro = CHydro # GW, GWh
        
        self.windFrag = np.ones(windFrag.shape[0])
        self.windFrag[stormZone] = windFrag[stormZone]

        self.stormDur = np.zeros(stormDur.shape[0])
        self.stormDur[stormZone] = stormDur[stormZone]

        self.stormDur = self.stormDur.astype(int)

        self.CWindR = self.CWind*(self.windFrag)
        self.GWindR = TSWind * np.tile(self.CWindR, (intervals, 1)) * pow(10, 3) # GWind(i, t), GW to MW
        self.WindDiff =  self.GWindR[:,stormZone] - self.GWind[:,stormZone]
        

        # self.LossR = self.GWind.sum(axis=1) - self.GWindR.sum(axis=1)        

        self.StormDeficit, self.cost, self.penalties = cost(self)

        # self.fragility = self.StormDeficit/(a constant?)

    def __repr__(self):
        """S = Solution(list(np.ones(64))) >> print(S)"""
        return 'Solution({})'.format(self.x)
    

