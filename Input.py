# Modelling input and assumptions
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

import numpy as np
import pandas as pd
from Optimisation import scenario, costConstraintFactor, eventZone, n_year, x0mode, event, trial, logic
from Simulation import Reliability
from CoSimulation import Resilience
from Network import Transmission
#%%
if trial is None: 
    suffix = '{}-{}-{}-{}.csv'.format(scenario, eventZone, n_year, str(event)[0])
else:
    suffix = '{}-{}-{}-{}-{}.csv'.format(scenario, eventZone, n_year, str(event)[0], trial)

if isinstance(eventZone, str):
    if eventZone.lower() == 'all': eventZone = 'All'
    elif eventZone.lower() == 'none': eventZone = 'None'
    else: raise Exception('eventZone not valid, when type str ("all" or None")')

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

if event in ('storm', 'drought'):
    durations = np.genfromtxt(f'Data/{event}Durations.csv', delimiter=',', usecols=[0,1])
    durations = durations[durations[:, 0].argsort()]
    durations = np.repeat(durations[:,0], durations[:,1].astype(int))
    
    coverage = np.genfromtxt(f'Data/{event}Coverage.csv')
    eventsPerYear = len(durations)/coverage
    percentile = 1/(eventsPerYear*n_year)
    
    durations = durations[-int(percentile*len(durations)):1-int(percentile*len(durations))]
        
    eventDur = np.repeat(int(durations[0]*2), len(Windl))
    
elif event == 'event':
    coverage = np.genfromtxt('Data/stormCoverage.csv'), np.genfromtxt('Data/droughtCoverage.csv')
    eventsPerYear = 0 
    durations = [np.genfromtxt('Data/stormDurations.csv', delimiter=',', usecols=[0,1]), 
                 np.genfromtxt('Data/droughtDurations.csv', delimiter=',', usecols=[0,1])]
    for i, dur in enumerate(durations): 
        dur = dur[dur[:,0].argsort()]
        dur = np.repeat(dur[:,0], dur[:,1].astype(int))

        eventsPerYear += len(dur)/coverage[i]
        durations[i] = dur
    durations = np.concatenate(durations)
    durations = durations[durations.argsort()]
    percentile = 1/(eventsPerYear*n_year)
    rank = int(percentile*len(durations))
    
    durations = durations[-rank:1-rank]
    eventDur = np.repeat(int(durations[0]*2), len(Windl))


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
    eventDur = eventDur[np.where(Windl==node)[0]]
    
    CHydro, CBio, CBaseload, CPeak, CDP, CDS = [x[np.where(Nodel==node)[0]] for x in (CHydro, CBio, CBaseload, CPeak, CDP, CDS)]
    if node=='QLD':
        MLoad, MLoadD, CDP, CDS = [x / 0.9 for x in (MLoad, MLoadD, CDP, CDS)]

    if isinstance(eventZone, np.ndarray): 
        zones = np.zeros(Windl.shape)
        zones[eventZone] = 1 
        zones = zones[np.where(Windl==node)[0]]
        eventZoneIndx = np.where(zones==1)[0]

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
    eventDur = eventDur[np.where(np.in1d(Windl, coverage)==True)[0]]
    
    CHydro, CBio, CBaseload, CPeak, CDP, CDS = [x[np.where(np.in1d(Nodel, coverage)==True)[0]] for x in (CHydro, CBio, CBaseload, CPeak, CDP, CDS)]
    if 'FNQ' not in coverage:
        MLoad[:, np.where(coverage=='QLD')[0][0]] /= 0.9
        MLoadD[:, np.where(coverage=='QLD')[0][0]] /= 0.9
        CDP[np.where(coverage == 'QLD')[0]] /= 0.9
        CDS[np.where(coverage == 'QLD')[0]] /= 0.9

    if isinstance(eventZone, np.ndarray): 
        zones = np.zeros(Windl.shape)
        zones[eventZone] = 1 
        zones = zones[np.where(np.in1d(Windl, coverage)==True)[0]]
        eventZoneIndx = np.where(zones==1)[0]

    Nodel, PVl, Windl = [x[np.where(np.in1d(x, coverage)==True)[0]] for x in (Nodel, PVl, Windl)]

intervals, nodes = MLoad.shape
years = int(resolution * intervals / 8760)
pzones, wzones = (TSPV.shape[1], TSWind.shape[1])
pidx, widx, sidx = (pzones, pzones + wzones, pzones + wzones + nodes)

energy = (MLoad + MLoadD).sum() * pow(10, -9) * resolution / years # PWh p.a.
contingency = list(0.25 * (MLoad + MLoadD).max(axis=0) * pow(10, -3)) # MW to GW

GBaseload = np.tile(CBaseload, (intervals, 1)) * pow(10, 3) # GW to MW

x0 = None
if x0mode == 3: 
    x0 = np.genfromtxt('CostOptimisationResults/Optimisation_resultx21-[14]-25-d.csv', delimiter=',')
if x0mode == 2: 
    try: 
        if trial is None: x0 = np.genfromtxt('Results/Optimisation_resultx{}-{}-{}-{}.csv'.format(scenario, eventZone, n_year, str(event)[0]), delimiter = ',')
        else: x0 = np.genfromtxt('Results/Optimisation_resultx{}-{}-{}-{}-{}.csv'.format(scenario, eventZone, n_year, str(event)[0], trial), delimiter = ',')
    except FileNotFoundError: pass 
if x0 is None and x0mode >= 1:
    try: x0 = np.genfromtxt('CostOptimisationResults/Optimisation_resultx{}-None.csv'.format(scenario), delimiter = ',')
    except FileNotFoundError: pass

if costConstraintFactor is not None:
    OptimisedCost = pd.read_csv('CostOptimisationResults/Costs.csv', index_col='Scenario').loc[scenario, 'LCOE']
    costConstraint = costConstraintFactor*OptimisedCost
else: 
    costConstraintFactor = np.inf
#%%
def cost(solution): 

    Deficit, DeficitD = Reliability(solution, flexible=np.zeros(intervals), output=True) # solutionj-EDE(t, j), MW
    Flexible = (Deficit + DeficitD / efficiencyD).sum() * resolution / years / (0.5 * (1 + efficiency)) # MWh p.a.
    Hydro = Flexible + GBaseload.sum() * resolution / years # Hydropower & biomass: MWh p.a.
    PenHydro = max(0, Hydro - 20 * pow(10, 6)) # TWh p.a. to MWh p.a.

    Deficit, DeficitD, RDeficit, RDeficitD = Resilience(solution, flexible=np.ones(intervals) * CPeak.sum() * pow(10, 3)) # solutionj-EDE(t, j), GW to MW
    PenDeficit = max(0, (Deficit + DeficitD / efficiencyD).sum() * resolution) # MWh
    eventDeficit = max(0, (RDeficit + RDeficitD / efficiencyD).sum() * resolution) #MWh

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

    return eventDeficit, LCOE, penalties


#%%
class Solution:
    """A candidate solution of decision variables CPV(i), CWind(i), CPHP(j), S-CPHS(j)"""

    def __init__(self, x):
        self.x = x
        
        self.logic = logic
        if isinstance(eventZone, str):
            if eventZone == 'All': self.eventZone, self.eventZoneIndx = 'All', np.arange(wzones)
            elif eventZone == 'None': self.eventZone, self.eventZoneIndx = 'None', None                
        elif isinstance(eventZone, np.ndarray): self.eventZone, self.eventZoneIndx  = eventZone, eventZoneIndx
        else: raise ValueError('eventZone should be "None", "All", or np array') 
        self.intervals = intervals
        
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
        
        self.eventDur = np.zeros(eventDur.shape)
        
        if self.eventZoneIndx is not None:
            
            self.eventDur[self.eventZoneIndx] = eventDur[self.eventZoneIndx]
        
        self.eventDur = np.rint(self.eventDur).astype(int)
        if len(eventZoneIndx) > 1 and logic == 'and': 
            assert len(np.unique(self.eventDur[np.where(self.eventDur != 0)])) == 1
        
        self.CWindR = np.array(self.CWind)*~self.eventDur.astype(bool)
        self.TSWindR = np.zeros(TSWind.shape)
        self.TSWindR[:, ~self.eventDur.astype(bool)] = TSWind[:, ~self.eventDur.astype(bool)]
        self.GWindR = self.TSWindR * np.tile(self.CWindR, (intervals, 1)) * pow(10, 3) # GWind(i, t), GW to MW
        self.WindDiff = self.GWindR-self.GWind
        
        self.OptimisedCost, self.costConstraint = OptimisedCost, costConstraint

        self.eventDeficit, self.cost, self.penalties = cost(self)
        

    def __repr__(self):
        """S = Solution(list(np.ones(64))) >> print(S)"""
        return 'Solution({})'.format(self.x)
    

#%%
if __name__ == '__main__': 
    s = Solution(np.genfromtxt(r"C:\Users\u6942852\Documents\Repos\FIRM_Australia\CostOptimisationResults\Optimisation_resultx21-None.csv", delimiter=','))
    Deficit, DeficitD, RDeficit, RDeficitD = Resilience(s, flexible=np.ones(intervals) * CPeak.sum() * pow(10, 3))
    print(RDeficit.sum())
    
    