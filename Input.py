# Modelling input and assumptions
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

import numpy as np
import pandas as pd
from Optimisation import scenario, costConstraintFactor, eventZone, n_year, x0mode, event, trial, logic, testing
from Simulation import Reliability
from CoSimulation import Resilience
from Network import Transmission
from numba import njit, float64, int64, prange, boolean, types
from numba.experimental import jitclass

#%%

Nodel = np.array(['FNQ', 'NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC', 'WA'])
PVl   = np.array(['NSW']*7 + ['FNQ']*1 + ['QLD']*2 + ['FNQ']*3 + ['SA']*6 + ['TAS']*0 + ['VIC']*1 + ['WA']*1 + ['NT']*1)
Windl = np.array(['NSW']*8 + ['FNQ']*1 + ['QLD']*2 + ['FNQ']*2 + ['SA']*8 + ['TAS']*4 + ['VIC']*4 + ['WA']*3 + ['NT']*1)
resolution = 0.5

_, Nodel_int = np.unique(Nodel, return_inverse=True)
_, PVl_int = np.unique(Nodel, return_inverse=True)
_, Windl_int = np.unique(Nodel, return_inverse=True)

if isinstance(eventZone, str):
    if eventZone.lower() == 'all': 
        eventZone = 'All'
    elif eventZone.lower() == 'none': 
        eventZone = 'None'
    else: 
        raise Exception('eventZone not valid, when type str ("all" or None")')

if trial is None: 
    suffix = '{}-{}-{}-{}.csv'.format(scenario, eventZone, n_year, str(event)[0])
else:
    suffix = '{}-{}-{}-{}-{}.csv'.format(scenario, eventZone, n_year, str(event)[0], trial) 


MLoad = np.genfromtxt('Data/electricity.csv', delimiter=',', skip_header=1, usecols=range(4, 4+len(Nodel))) # EOLoad(t, j), MW
for i in ['evan', 'erigid', 'earticulated', 'enonfreight', 'ebus', 'emotorcycle', 'erail', 'eair', 'ewater', 'ecooking', 'emanufacturing', 'emining']:
    MLoad += np.genfromtxt('Data/{}.csv'.format(i), delimiter=',', skip_header=1, usecols=range(4, 4+len(Nodel)))

DSP = 0.8 if scenario>=31 else 0
MLoad += (1 - DSP) * np.genfromtxt('Data/ecar.csv', delimiter=',', skip_header=1, usecols=range(4, 4+len(Nodel)))

MLoadD = DSP * np.genfromtxt('Data/ecar.csv', delimiter=',', skip_header=1, usecols=range(4, 4+len(Nodel)))

TSPV = np.genfromtxt('Data/pv.csv', delimiter=',', skip_header=1, usecols=range(4, 4+len(PVl))) # TSPV(t, i), MW
TSWind = np.genfromtxt('Data/wind.csv', delimiter=',', skip_header=1, usecols=range(4, 4+len(Windl))) # TSWind(t, i), MW

if testing is True: 
    MLoad  = MLoad[ :3*30*24*2,:]
    MLoadD = MLoadD[:3*30*24*2,:]
    TSPV   = TSPV[  :3*30*24*2,:]
    TSWind = TSWind[:3*30*24*2,:]

if event in ('storm', 'drought'):
    durations = np.genfromtxt(f'Data/{event}Durations.csv', delimiter=',', usecols=[0,1])
    durations = durations[durations[:, 0].argsort()]
    durations = np.repeat(durations[:,0], durations[:,1].astype(int))
    
    coverage = np.genfromtxt(f'Data/{event}Coverage.csv')
    eventsPerYear = len(durations)/coverage
    percentile = 1/(eventsPerYear*n_year)
    
    durations = durations[-int(percentile*len(durations)):1-int(percentile*len(durations))]
        
    eventDur = int(durations[0] / resolution)
    
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
    eventDur = int(durations[0] / resolution)
else: 
    assert event is None 
    eventDur = 0

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
finalyear = firstyear if testing else finalyear

if scenario<=17:
    node = Nodel[scenario % 10]

    MLoad, MLoadD = [x[:, np.where(Nodel==node)[0]] for x in (MLoad, MLoadD)]
    TSPV = TSPV[:, np.where(PVl==node)[0]]
    TSWind = TSWind[:, np.where(Windl==node)[0]]
    
    CHydro, CBio, CBaseload, CPeak, CDP, CDS = [x[np.where(Nodel==node)[0]] for x in (CHydro, CBio, CBaseload, CPeak, CDP, CDS)]
    if node=='QLD':
        MLoad, MLoadD, CDP, CDS = [x / 0.9 for x in (MLoad, MLoadD, CDP, CDS)]

    if isinstance(eventZone, np.ndarray): 
        zones = np.zeros(Windl.shape)
        zones[eventZone] = 1 
        zones = zones[np.where(Windl==node)[0]]
        eventZoneIndx = np.where(zones==1)[0]
    else:
        eventZoneIndx = None
        
    Nodel_int, PVl_int, Windl_int = [x[np.where(Nodel==node)[0]] for x in (Nodel_int, PVl_int, Windl_int)]
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
    else:
        eventZoneIndx = None
        
    Nodel_int, PVl_int, Windl_int = [x[np.where(np.in1d(Nodel, coverage)==True)[0]] for x in (Nodel_int, PVl_int, Windl_int)]
    Nodel, PVl, Windl = [x[np.where(np.in1d(x, coverage)==True)[0]] for x in (Nodel, PVl, Windl)]

intervals, nodes = MLoad.shape
if testing is True:
    years = resolution * intervals / 8760
else: 
    years = int(resolution * intervals / 8760)
pzones, wzones = (TSPV.shape[1], TSWind.shape[1])
pidx, widx, sidx = (pzones, pzones + wzones, pzones + wzones + nodes)

energy = (MLoad + MLoadD).sum() * pow(10, -9) * resolution / years # PWh p.a.
contingency = list(0.25 * (MLoad + MLoadD).max(axis=0) * pow(10, -3)) # MW to GW

GBaseload = np.tile(CBaseload, (intervals, 1)) * pow(10, 3) # GW to MW

x0 = None
if x0mode == 2: 
    try: 
        x0 = np.genfromtxt(f'Results/Optimisation_resultx{suffix}.csv', delimiter = ',')
    except FileNotFoundError: 
        pass 
if x0 is None and x0mode >= 1:
    try: 
        x0 = np.genfromtxt('CostOptimisationResults/Optimisation_resultx{}-None.csv'.format(scenario), delimiter = ',')
    except FileNotFoundError: 
        pass

if costConstraintFactor is not None:
    OptimisedCost = pd.read_csv('CostOptimisationResults/Costs.csv', index_col='Scenario').loc[scenario, 'LCOE']
    costConstraint = costConstraintFactor*OptimisedCost
else: 
    costConstraintFactor = np.inf

if isinstance(eventZone, str):
    if eventZone == 'All': 
        eventZoneIndx = np.arange(wzones, dtype=np.int64)
    elif eventZone == 'None': 
        eventZoneIndx = np.array([-1], dtype=np.int64)  
elif isinstance(eventZone, np.ndarray): 
    eventZoneIndx = eventZoneIndx.astype(np.int64)
elif eventZone is None:
    eventZoneIndx = np.array([-1], dtype=np.int64)                     
else: 
    raise ValueError('eventZone should be None, "None", "All", or np array') 

lb = np.array([0.]  * pzones + [0.]   * wzones + contingency   + [0.])
ub = np.array([50.] * pzones + [50.]  * wzones + [50.] * nodes + [5000.])

#%%
@njit()
def F(solution): 
    Deficit, DeficitD = Reliability(solution, flexible=np.zeros((intervals, ) , dtype=np.float64)) # Sj-EDE(t, j), MW
    Flexible = Deficit.sum() * resolution / years / (0.5 * (1 + efficiency)) # MWh p.a.
    Hydro = Flexible + GBaseload.sum() * resolution / years # Hydropower & biomass: MWh p.a.
    PenHydro = np.maximum(0, Hydro - 20 * 1000000) # TWh p.a. to MWh p.a.

    Deficit, DeficitD, RDeficit, RDeficitD = Resilience(solution, flexible=np.ones(intervals, dtype=np.float64) * CPeak.sum() * 1000) # solutionj-EDE(t, j), GW to MW
    RDeficit, RDeficitD = RDeficit / len(solution.eventZoneIndx), RDeficitD / len(solution.eventZoneIndx)
    PenDeficit = max(0, (Deficit + DeficitD / efficiencyD).sum() * resolution) # MWh
    eventDeficit = max(0, (RDeficit + RDeficitD / efficiencyD).sum() * resolution / (intervals*resolution/24)) #MWh/day

    TDC_abs = np.abs(Transmission(solution)) if scenario>=21 else np.zeros((intervals, len(DCloss))) # TDC: TDC(t, k), MW
    CDC = np.zeros(len(DCloss), dtype=np.float64)
    for j in prange(len(DCloss)):
        for i in range(intervals):
            CDC[j] = np.maximum(TDC_abs[i, j], CDC[j])
    CDC = CDC * 0.001 # CDC(k), MW to GW
    PenDC = max(0, CDC[6] - CDC6max) * 0.001 # GW to MW

    cost = factor * np.array([solution.CPV.sum(), solution.CWind.sum(), solution.CPHP.sum(), solution.CPHS] 
                             + list(CDC) + [solution.CPV.sum(), solution.CWind.sum(), Hydro * 0.000_001, -1, -1])
    if scenario<=17:
        cost[-1], cost[-2] = 0,0
        
    loss = np.sum(TDC_abs, axis=0) * DCloss
    loss = loss.sum() * 0.000000001 * resolution / years # PWh p.a.
    LCOE = cost.sum() / abs(energy - loss)

    penalties = PenHydro + PenDeficit + PenDC

    return eventDeficit, LCOE, penalties

#%%
# Specify the types for jitclass
solution_spec = [
    ('x', float64[:]),  # x is 1d array
    ('scenario', int64),
    ('intervals', int64),
    ('nodes', int64),
    ('resolution',float64),
    ('Nodel_int', int64[:]),  # 1D array of floats
    ('PVl_int', int64[:]),
    ('Windl_int', int64[:]),

    ('CPV', float64[:]),
    ('CWind', float64[:]),
    ('CWindR', float64[:]),
    ('GPV', float64[:, :]),  # 2D array of floats
    ('GWind', float64[:, :]),
    ('CPHP', float64[:]),
    ('CPHS', float64),
    ('CDP', float64[:]),
    ('CDS', float64[:]),
    ('efficiency', float64),
    ('efficiencyD', float64),

    ('MLoad', float64[:, :]),
    ('MLoadD', float64[:, :]),
    ('GBaseload', float64[:, :]),  
    ('CPeak', float64[:]),
    ('CHydro', float64[:]),
    
    ('logic', types.unicode_type),
    ('eventDur', int64),
    ('eventZoneIndx', int64[:]),
    
    ('GWindR', float64[:,:]),
    ('WindDiff', float64[:,:]),
    
    ('flexible', float64[:]),
    ('Discharge', float64[:]),
    ('Charge', float64[:]),
    ('Storage', float64[:]),
    ('Deficit', float64[:]),
    ('DischargeD', float64[:]),
    ('ChargeD', float64[:]),
    ('StorageD', float64[:]),
    ('DeficitD', float64[:]),
    ('Spillage', float64[:]),
    ('P2V', float64[:]),

    ('RDischarge', float64[:,:]),
    ('RCharge', float64[:,:]),
    ('RStorage', float64[:,:]),
    ('RDeficit', float64[:,:]),
    ('RDischargeD', float64[:,:]),
    ('RChargeD', float64[:,:]),
    ('RStorageD', float64[:,:]),
    ('RDeficitD', float64[:,:]),
    ('RSpillage', float64[:,:]),
    ('RP2V', float64[:,:]),
    ('RFlexible',float64[:]),

    ('MWindR', float64[:, :]),
    ('MPeakR', float64[:, :]),
    ('MHydroR', float64[:, :]),
    ('MBioR', float64[:, :]),
    
    ('OptimisedCost', float64),
    ('costConstraint', float64),
    ('eventDeficit', float64),
    ('cost', float64),
    ('penalties', float64),
    ('evaluated', boolean),
    
    ('MBaseload', float64[:, :]),
    ('MPV', float64[:, :]),
    ('MP2V', float64[:, :]),
    ('MWind', float64[:, :]),
    ('MPeak', float64[:, :]),
    ('MDischarge', float64[:, :]),
    ('MCharge', float64[:, :]),
    ('MStorage', float64[:, :]),
    ('MDischargeD', float64[:, :]),
    ('MChargeD', float64[:, :]),
    ('MStorageD', float64[:, :]),
    ('MDeficit', float64[:, :]),
    ('MSpillage', float64[:, :]),
    ('MHydro', float64[:, :]),
    ('MBio', float64[:, :]),
    
    ('TDC', float64[:, :]),
    ('CDC', float64[:]),
    ('Topology', float64[:, :]),
    ('FQ', float64[:]),
    ('NQ', float64[:]),
    ('NS', float64[:]),
    ('NV', float64[:]),
    ('AS', float64[:]),
    ('SW', float64[:]),
    ('TV', float64[:]),
]

@jitclass(solution_spec)
class Solution:
    """A candidate solution of decision variables CPV(i), CWind(i), CPHP(j), S-CPHS(j)"""

    def __init__(self, x):
        self.x = x
        self.scenario = scenario
        self.intervals, self.nodes = (intervals, nodes)
        self.resolution = resolution
        self.Nodel_int, self.PVl_int, self.Windl_int = (Nodel_int, PVl_int, Windl_int)

        self.CPV = x[: pidx] # CPV(i), GW
        self.CWind = x[pidx: widx] # CWind(i), GW        
        self.CPHP = x[widx: sidx] # CPHP(j), GW
        self.CPHS = x[sidx] # S-CPHS(j), GWh
        self.CDP, self.CDS = (CDP, CDS)  # GW, GWh
        self.efficiency, self.efficiencyD = (efficiency, efficiencyD)

        # Manually replicating np.tile functionality for CPV and CWind
        CPV_tiled = np.zeros((intervals, len(self.CPV)))
        CWind_tiled = np.zeros((intervals, len(self.CWind)))
        for i in range(intervals):
            for j in range(len(self.CPV)):
                CPV_tiled[i, j] = self.CPV[j]
            for j in range(len(self.CWind)):
                CWind_tiled[i, j] = self.CWind[j]

        self.GPV = TSPV * CPV_tiled * 1000.  # GPV(i, t), GW to MW
        self.GWind = TSWind * CWind_tiled * 1000.  # GWind(i, t), GW to MW
        self.MLoad, self.MLoadD = (MLoad, MLoadD)
        self.GBaseload, self.CPeak = (GBaseload, CPeak)
        self.CHydro = CHydro # GW, GWh
        
        self.logic = logic        
        self.eventZoneIndx, self.eventDur = eventZoneIndx, eventDur
        
        mask = np.ones(self.CWind.shape, np.bool_)
        if eventZoneIndx[0] >= 0:
            mask[self.eventZoneIndx] = 0 
        
        self.CWindR = self.CWind * mask
        TSWindR = TSWind * mask
        
        CWindR_tiled = np.zeros((intervals, len(self.CWindR)), dtype=np.float64)
        for i in range(intervals):
            for j in range(len(self.CWindR)):
                CWindR_tiled[i,j] = self.CWindR[j]
        self.GWindR = TSWindR * CWindR_tiled * 1000 # GWind(i, t), GW to MW
        
        self.WindDiff = self.GWindR - self.GWind
        
        self.OptimisedCost, self.costConstraint = OptimisedCost, costConstraint
        
        self.evaluated=False
        
        
    def _evaluate(self):
        self.eventDeficit, self.cost, self.penalties = F(self)
        self.evaluated=True

    # def __repr__(self):
    #     """S = Solution(list(np.ones(64))) >> print(S)"""
    #     return 'Solution({})'.format(self.x)
    

#%%
if __name__ == '__main__': 
    x = np.genfromtxt(f"CostOptimisationResults/Optimisation_resultx{scenario}-None.csv", delimiter=',')
    # x = np.genfromtxt(f"Results/Optimisation_resultx{scenario}-None-25-e.csv", delimiter=',')
    # x = np.genfromtxt(f"Results/Optimisation_resultx{scenario}-None-25-N-0.csv", delimiter=',')
    solution = Solution(x)
    solution._evaluate()
    print(solution.eventDeficit, solution.cost, solution.penalties)
    # Deficit, DeficitD, RDeficit, RDeficitD = Resilience(s, flexible=np.ones(intervals) * CPeak.sum() * 1000)
    # print(RDeficit.sum())
    
    