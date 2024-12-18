# Modelling input and assumptions
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

import numpy as np
from numba import njit, prange, float64, int64, boolean
from numba.experimental import jitclass
from argparse import ArgumentParser

from Simulation import Reliability
from Network import Transmission
from Costs import *

parser = ArgumentParser()
parser.add_argument('-i', default=400, type=int, required=False, help='maxiter=4000, 400')
parser.add_argument('-p', default=1, type=int, required=False, help='popsize=2, 10')
parser.add_argument('-m', default=0.5, type=float, required=False, help='mutation=0.5')
parser.add_argument('-r', default=0.3, type=float, required=False, help='recombination=0.3')
parser.add_argument('-s', default=21, type=int, required=False, help='11, 12, 13, ...')
args = parser.parse_args()

scenario = args.s

Nodel = np.array(['FNQ', 'NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC', 'WA'])
PVl =   np.array(['NSW']*9 + ['FNQ']*5 + ['QLD']*4 + ['SA']*9 + ['TAS']*3 + ['VIC']*6)
OnsWl = np.array(['NSW']*9 + ['FNQ']*5 + ['QLD']*4 + ['SA']*9 + ['TAS']*3 + ['VIC']*6)
OffWl = np.array(['NSW']*2 + ['SA']*1 + ['TAS']*2 + ['VIC']*2)
resolution = 0.5

n_node = dict((name, i) for i, name in enumerate(Nodel))

# NT and WA not currently supported. To be included in future update
nodesupportl = np.array(['FNQ', 'NSW', 'QLD', 'SA', 'TAS', 'VIC'])
nodesupport = len(np.setdiff1d(Nodel, nodesupportl))

MLoad = np.genfromtxt('Data/electricity.csv', delimiter=',', skip_header=1, usecols=range(4, 4+len(Nodel)-nodesupport)) # EOLoad(t, j), MW
#behind the meter solar 
MLoad -= np.genfromtxt('Data/non-scheduled_pv.csv', delimiter=',', skip_header=1, usecols=range(4, 4+len(Nodel)-nodesupport))

TSPV    = np.genfromtxt('Data/utility_pv.csv',   delimiter=',', skip_header=1, usecols=range(4, 4+len(PVl))) # TSPV(t, i), MW
TSOnsW  = np.genfromtxt('Data/onshore_high.csv', delimiter=',', skip_header=1, usecols=range(4, 4+len(OnsWl))) # TSWind(t, i), MW
TSOffW = np.genfromtxt('Data/offshore_fixed.csv', delimiter=',', skip_header=1, usecols=range(4, 4+len(OffWl))) # TSWind(t, i), MW

assets = np.genfromtxt('Data/hydrobio.csv', dtype=None, delimiter=',', encoding=None)[1:, 1:].astype(float)
CHydro, CBio = [assets[:, x] * pow(10, -3) for x in range(assets.shape[1])] # CHydro(j), MW to GW
CBaseload = np.array([0, 0, 0, 0, 0, 1.0, 0, 0]) # 24/7, GW
CPeak = CHydro + CBio - CBaseload # GW

DClengths = np.array([1500, 1000, 1000, 800, 1200, 2400, 400]) 
DCloss = DClengths * 0.03 * pow(10, -3)
undersea_mask = np.array([0, 0, 0, 0, 0, 0, 1], dtype=bool)
# FQ, NQ, NS, NV, AS, SW, only TV constrained
CDC6max = 3 * 0.63 # GW

efficiency = 0.8
firstyear, finalyear, timestep = (2025, 2034, 1)

if scenario<=17:
    node = Nodel[scenario % 10]

    MLoad  = MLoad[:,  np.where(Nodel==node)[0]]
    TSPV   = TSPV[:,   np.where(PVl  ==node)[0]]
    TSOnsW = TSOnsW[:, np.where(OnsWl==node)[0]]
    TSOffW = TSOffW[:, np.where(OffWl==node)[0]]
    CHydro, CBio, CBaseload, CPeak = [x[np.where(Nodel==node)[0]] for x in (CHydro, CBio, CBaseload, CPeak)]
    if node=='QLD':
        MLoad /= 0.9 

    Nodel, PVl, OnsWl, OffWl = [x[np.where(x==node)[0]] for x in (Nodel, PVl, OnsWl, OffWl)]

if scenario>=21:
    Nodel = nodesupportl
    coverage = [np.array(['NSW', 'QLD', 'SA', 'TAS', 'VIC']),
                np.array(['NSW', 'QLD', 'SA', 'TAS', 'VIC', 'WA']),
                np.array(['NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC']),
                np.array(['NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC', 'WA']),
                np.array(['FNQ', 'NSW', 'QLD', 'SA', 'TAS', 'VIC']),
                np.array(['FNQ', 'NSW', 'QLD', 'SA', 'TAS', 'VIC', 'WA']),
                np.array(['FNQ', 'NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC']),
                np.array(['FNQ', 'NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC', 'WA'])][scenario % 10 - 1]

    MLoad  = MLoad[:,  np.where(np.in1d(Nodel, coverage))[0]]
    TSPV   = TSPV[:,   np.where(np.in1d(PVl,   coverage))[0]]
    TSOnsW = TSOnsW[:, np.where(np.in1d(OnsWl, coverage))[0]]
    TSOffW = TSOffW[:, np.where(np.in1d(OffWl, coverage))[0]]
    CHydro, CBio, CBaseload, CPeak = [x[np.where(np.in1d(Nodel, coverage)==True)[0]] for x in (CHydro, CBio, CBaseload, CPeak)]
    if 'FNQ' not in coverage:
        MLoad[:, np.where(coverage=='QLD')[0][0]] /= 0.9

    Nodel, PVl, OnsWl, OffWl = [x[np.where(np.in1d(x, coverage)==True)[0]] for x in (Nodel, PVl, OnsWl, OffWl)]

if 'WA' in Nodel or 'NT' in Nodel:
    raise NotImplementedError("Try a different scenario")

Nodel_int, PVl_int, OnsWl_int, OffWl_int = (np.array([n_node[node] for node in x], dtype=np.int64) for x in (Nodel, PVl, OnsWl, OffWl))

intervals, nodes = MLoad.shape
nhvdc = len(DCloss)
years = int(resolution * intervals / 8760)
pvzones, onswzones, offwzones = TSPV.shape[1], TSOnsW.shape[1], TSOffW.shape[1]
pvidx   = pvzones
onswidx = pvidx   + onswzones
offwidx = onswidx + offwzones
sidx    = offwidx + nodes

energy = MLoad.sum() * resolution / years # MWh p.a.
contingency = list(0.25 * MLoad.max(axis=0) * pow(10, -3)) # MW to GW

GBaseload = np.tile(CBaseload, (intervals, 1)) * pow(10, 3) # GW to MW

lb = np.array([0.]  * pvzones + [0.]  * (onswzones+offwzones) + contingency   + [0.])
ub = np.array([50.] * pvzones + [50.] * (onswzones+offwzones) + [50.] * nodes + [5000.])

solution_spec = [
    ('x',           float64[:]      ),  
    ('scenario',    int64           ),  
    ('intervals',   int64           ),
    ('nodes',       int64           ),
    ('resolution',  float64         ),
    ('efficiency',  float64         ),
    ('Nodel_int',   int64[:]        ), 
    ('PVl_int',     int64[:]        ),
    ('OnsWl_int',   int64[:]        ),
    ('OffWl_int',   int64[:]        ),
    ('CPV',         float64[:]      ), 
    ('COnsW',       float64[:]      ), 
    ('COffW',       float64[:]      ), 
    ('CPHP',        float64[:,]     ),
    ('CPHS',        float64         ),
    ('CPeak',       float64[:]      ), 
    ('CHydro',      float64[:]      ), 
    ('GPV',         float64[:, :]   ),  
    ('GOnsW',       float64[:, :]   ),  
    ('GOffW',       float64[:, :]   ),  
    ('GBaseload',   float64[:, :]   ),
    ('flexible',    float64[:]      ),
    ('Discharge',   float64[:]      ),
    ('Charge',      float64[:]      ),
    ('Storage',     float64[:]      ),
    ('Deficit',     float64[:]      ),
    ('Spillage',    float64[:]      ),
    ('Penalties',   float64         ),
    ('LCOE',        float64         ),
    ('LCOG',        float64         ),
    ('LCOBS',       float64         ),
    ('LCOBT',       float64         ),
    ('LCOBL',       float64         ),
    ('MLoad',       float64[:, :]   ),  
    ('MPV',         float64[:, :]   ),
    ('MOnsW',       float64[:, :]   ),
    ('MOffW',       float64[:, :]   ),
    ('MPeak',       float64[:, :]   ),
    ('MHydro',      float64[:, :]   ),
    ('MBio',        float64[:, :]   ),
    ('MDischarge',  float64[:, :]   ),
    ('MCharge',     float64[:, :]   ),
    ('MStorage',    float64[:, :]   ),
    ('MDeficit',    float64[:, :]   ),
    ('MSpillage',   float64[:, :]   ),
    ('MImport',     float64[:, :]   ),
    ('TDC',         float64[:, :]   ),
    ('CDC',         float64[:]      ),
    ('FQ',          float64[:]      ),
    ('NQ',          float64[:]      ),
    ('NS',          float64[:]      ),
    ('NV',          float64[:]      ),
    ('AS',          float64[:]      ),
    ('SW',          float64[:]      ),
    ('TV',          float64[:]      ),
]

@jitclass(solution_spec)
class Solution:
    """A candidate solution of decision variables CPV(i), CWind(i), CPHP(j), S-CPHS(j)"""

    def __init__(self, x):
        self.x = x
        self.MLoad = MLoad
        self.intervals, self.nodes = (intervals, nodes)
        self.resolution = resolution

        self.CPV   = x[       : pvidx ]
        self.COnsW = x[pvidx  : onswidx]
        self.COffW = x[onswidx: offwidx]
        self.CPHP  = x[offwidx: sidx]
        self.CPHS  = x[sidx] 
        
        self.GPV   = TSPV   * np.ones((intervals, len(self.CPV  ))) * self.CPV * 1000. 
        self.GOnsW = TSOnsW * np.ones((intervals, len(self.COnsW))) * self.COnsW * 1000. 
        self.GOffW = TSOffW * np.ones((intervals, len(self.COffW))) * self.COffW * 1000. 

        self.efficiency = efficiency

        self.Nodel_int, self.PVl_int, self.OnsWl_int, self.OffWl_int = Nodel_int, PVl_int, OnsWl_int, OffWl_int
        self.scenario = scenario

        self.GBaseload, self.CPeak = (GBaseload, CPeak)
        self.CHydro = CHydro # GW, GWh

flex_min = np.zeros(intervals, dtype=np.float64)
flex_max  = np.ones( intervals, dtype=np.float64)*CPeak.sum()*1000
GBase = GBaseload.sum()*resolution/years
TDC_empty = np.zeros((intervals, len(DCloss)), dtype=np.float64)
#%%
@njit 
def Objective(S):
    Hydro = GBase + Reliability(S, flexible=flex_min).sum() * resolution / years
    Penalties = max(0., Hydro - 20_000_000) #Hydro over capacity
    Penalties += max(0, Reliability(S, flexible=flex_max).sum() * resolution) #Deficit
    
    TDC = np.abs(Transmission(S))*0.001 if scenario>=21 else TDC_empty # TDC: TDC(t, k), MW
    CDC = np.zeros(len(DCloss), np.float64)
    for j in range(nhvdc):
        for i in range(intervals):
            CDC[j] = max(TDC[i, j], CDC[j])
    
    Penalties += max(0, CDC[6] - CDC6max) * pow(10, 3) # DCmax
    
    LCOE = ((
        + S.CPV.sum()   * (pv_costs   + ACgen_costs)
        + S.COnsW.sum() * (onsw_costs + ACgen_costs) 
        + S.COffW.sum() * (offw_costs + ACgen_costs)
        + S.CPHP.sum()  * phes_costs[0]
        + S.CPHS        * phes_costs[1] 
        + S.Discharge.sum()*resolution/years * phes_costs[2]
        + phes_costs[3] 
        + (CDC*hvdc_costs).sum()
        + Hydro * hydro_purchase
        ) / energy)
        # ) / abs(energy - (np.sum(TDC, axis=0) * DCloss).sum() * resolution / years))
    return LCOE + Penalties
    
if __name__ == '__main__':
    x = np.genfromtxt(f'Results/Optimisation_resultx{scenario}.csv', delimiter=',', dtype=float)
    S = Solution(x)
    print(Objective(S))
    