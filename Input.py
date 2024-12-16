# Modelling input and assumptions
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

import numpy as np
from argparse import ArgumentParser

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
OffsWl = np.array(['NSW']*2 + ['SA']*1 + ['TAS']*2 + ['VIC']*2)
resolution = 0.5

nodesupportl = np.array(['FNQ', 'NSW', 'QLD', 'SA', 'TAS', 'VIC'])
nodesupport = len(np.setdiff1d(Nodel, nodesupportl))

MLoad = np.genfromtxt('Data/electricity.csv', delimiter=',', skip_header=1, usecols=range(4, 4+len(Nodel)-nodesupport)) # EOLoad(t, j), MW
#behind the meter solar 
MLoad -= np.genfromtxt('Data/non-scheduled_pv.csv', delimiter=',', skip_header=1, usecols=range(4, 4+len(Nodel)-nodesupport))

TSPV    = np.genfromtxt('Data/utility_pv.csv',   delimiter=',', skip_header=1, usecols=range(4, 4+len(PVl))) # TSPV(t, i), MW
TSOnsW  = np.genfromtxt('Data/onshore_high.csv', delimiter=',', skip_header=1, usecols=range(4, 4+len(OnsWl))) # TSWind(t, i), MW
TSOffsW = np.genfromtxt('Data/offshore_fixed.csv', delimiter=',', skip_header=1, usecols=range(4, 4+len(OffsWl))) # TSWind(t, i), MW

assets = np.genfromtxt('Data/hydrobio.csv', dtype=None, delimiter=',', encoding=None)[1:, 1:].astype(float)
CHydro, CBio = [assets[:, x] * pow(10, -3) for x in range(assets.shape[1])] # CHydro(j), MW to GW
CBaseload = np.array([0, 0, 0, 0, 0, 1.0, 0, 0]) # 24/7, GW
CPeak = CHydro + CBio - CBaseload # GW

# FQ, NQ, NS, NV, AS, SW, only TV constrained
CDC6max = 3 * 0.63 # GW

DClengths = np.array([1500, 1000, 1000, 800, 1200, 2400, 400]) 
DCloss = DClengths * 0.03 * pow(10, -3)
undersea_mask = np.array([0, 0, 0, 0, 0, 0, 1], dtype=bool)

efficiency = 0.8
firstyear, finalyear, timestep = (2025, 2034, 1)

if scenario<=17:
    node = Nodel[scenario % 10]

    MLoad   = MLoad[:,   np.where(Nodel ==node)[0]]
    TSPV    = TSPV[:,    np.where(PVl   ==node)[0]]
    TSOnsW  = TSOnsW[:,  np.where(OnsWl ==node)[0]]
    TSOffsW = TSOffsW[:, np.where(OffsWl==node)[0]]
    CHydro, CBio, CBaseload, CPeak = [x[np.where(Nodel==node)[0]] for x in (CHydro, CBio, CBaseload, CPeak)]
    if node=='QLD':
        MLoad /= 0.9 

    Nodel, PVl, OnsWl, OffsWl = [x[np.where(x==node)[0]] for x in (Nodel, PVl, OnsWl, OffsWl)]

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

    MLoad   = MLoad[:,  np.where(np.in1d(Nodel,  coverage))[0]]
    TSPV    = TSPV[:,   np.where(np.in1d(PVl,    coverage))[0]]
    TSOnsW  = TSOnsW[:, np.where(np.in1d(OnsWl,  coverage))[0]]
    TSOffsW = TSOnsW[:, np.where(np.in1d(OffsWl, coverage))[0]]
    CHydro, CBio, CBaseload, CPeak = [x[np.where(np.in1d(Nodel, coverage)==True)[0]] for x in (CHydro, CBio, CBaseload, CPeak)]
    if 'FNQ' not in coverage:
        MLoad[:, np.where(coverage=='QLD')[0][0]] /= 0.9

    Nodel, PVl, OnsWl, OffsWl = [x[np.where(np.in1d(x, coverage)==True)[0]] for x in (Nodel, PVl, OnsWl, OffsWl)]

if 'WA' in Nodel or 'NT' in Nodel:
    raise NotImplementedError("Try a different scenario")

intervals, nodes = MLoad.shape
years = int(resolution * intervals / 8760)
pvzones, onswzones, offswzones = TSPV.shape[1], TSOnsW.shape[1], TSOffsW.shape[1]
pvidx   = pvzones
onswidx  = pvidx   + onswzones
offswidx = onswidx  + offswzones
sidx    = offswidx + nodes

energy = MLoad.sum() * pow(10, -9) * resolution / years # PWh p.a.
contingency = list(0.25 * MLoad.max(axis=0) * pow(10, -3)) # MW to GW

GBaseload = np.tile(CBaseload, (intervals, 1)) * pow(10, 3) # GW to MW

lb = np.array([0.]  * pvzones + [0.]  * (onswzones+offswzones) + contingency   + [0.])
ub = np.array([50.] * pvzones + [50.] * (onswzones+offswzones) + [50.] * nodes + [5000.])


class Solution:
    """A candidate solution of decision variables CPV(i), CWind(i), CPHP(j), S-CPHS(j)"""

    def __init__(self, x):
        self.x = x
        self.MLoad = MLoad
        self.intervals, self.nodes = (intervals, nodes)
        self.resolution = resolution

        self.CPV    = list(x[       : pvidx ]) # CPV(i), GW
        self.COnsW  = list(x[pvidx  : onswidx]) # CWind(i), GW
        self.COffsW = list(x[onswidx : offswidx]) # CWind(i), GW
        self.CPHP   = list(x[offswidx: sidx]) # CPHP(j), GW
        self.CPHS   = x[sidx] # S-CPHS(j), GWh
        
        self.GPV    = TSPV    * np.tile(self.CPV,    (intervals, 1)) * pow(10, 3) # GPV(i, t), GW to MW
        self.GOnsW  = TSOnsW  * np.tile(self.COnsW,  (intervals, 1)) * pow(10, 3) # GWind(i, t), GW to MW
        self.GOffsW = TSOffsW * np.tile(self.COffsW, (intervals, 1)) * pow(10, 3) # GWind(i, t), GW to MW


        self.efficiency = efficiency

        self.Nodel, self.PVl, self.OnsWl, self.OffsWl = Nodel, PVl, OnsWl, OffsWl
        self.scenario = scenario

        self.GBaseload, self.CPeak = (GBaseload, CPeak)
        self.CHydro = CHydro # GW, GWh

    def __repr__(self):
        """S = Solution(list(np.ones(64))) >> print(S)"""
        return 'Solution({})'.format(self.x)