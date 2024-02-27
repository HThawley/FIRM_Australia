# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 11:54:57 2024

@author: u6942852
"""

from argparse import ArgumentParser
import numpy as np
from multiprocessing import Pool, cpu_count
import pandas as pd
import csv

parser = ArgumentParser()
parser.add_argument('-i',  default=400,   type=int,  required=False, help='maxiter=4000, 400')
parser.add_argument('-p',  default=1,     type=int,  required=False, help='popsize=2, 10')
parser.add_argument('-m',  default=0.5,   type=float,required=False, help='mutation=0.5')
parser.add_argument('-r',  default=0.3,   type=float,required=False, help='recombination=0.3')
parser.add_argument('-s',  default=11,    type=int,  required=False, help='11, 12, 13, ...')
parser.add_argument('-his',default=1,     type=int,  required=False, help='save history')
parser.add_argument('-x',  default=0,     type=int,  required=False, help='first guess. 2=restart, 1=costOptimum, 0=random')
parser.add_argument('-v',  default=1,     type=int,  required=False, help='verbose - 1 for True, 0 for False')
parser.add_argument('-c',  default=np.inf,type=float,required=False, help='cost slack')
parser.add_argument('-n',  default=0,     type=int,  required=False, help='Number of alternatives to generate')
args = parser.parse_args()

scenario = args.s

from Input import *
from Simulation import Reliability
from Network import Transmission

lb = [0.]  * pzones + [0.]   * wzones + contingency   + [0.]
ub = [50.] * pzones + [50.]  * wzones + [50.] * nodes + [5000.]
bounds = list(zip(lb, ub))

if args.c is not None:
    optimisedCost = np.genfromtxt('CostOptimisationResults/Costs.csv', delimiter=',', skip_header=1)
    optimisedCost = optimisedCost[optimisedCost[:,0] == scenario, 1][0]
    costConstraint = args.c*optimisedCost
else: 
    costConstraint = np.inf

def penalties(x):
    with Pool(processes=cpu_count()) as processPool:
        arrs = [xn for xn in x.T] if len(x.shape) == 0 else [x]
        pHydro = processPool.map(penHydro, arrs)
        pDeficit = processPool.map(penDeficit, arrs)
        pDC = processPool.map(penDC, arrs)
    results = np.stack([np.array(pool) for pool in (pHydro, pDeficit, pDC)], axis = 1)
    return results.sum(axis=0)

def penHydro(x, S=None):
    S = Solution(x) if S is None else S
    Deficit, DeficitD = Reliability(S, flexible=np.zeros(intervals)) # Sj-EDE(t, j), MW
    Flexible = (Deficit + DeficitD / efficiencyD).sum() * resolution / years / (0.5 * (1 + efficiency)) # MWh p.a.
    Hydro = Flexible + GBaseload.sum() * resolution / years # Hydropower & biomass: MWh p.a.
    PenHydro = max(0, Hydro - 20 * pow(10, 6)) # TWh p.a. to MWh p.a.
    return PenHydro

def penDeficit(x, S=None):
    S = Solution(x) if S is None else S
    Deficit, DeficitD = Reliability(S, flexible=np.ones(intervals) * CPeak.sum() * pow(10, 3)) # Sj-EDE(t, j), GW to MW
    PenDeficit = max(0, (Deficit + DeficitD / efficiencyD).sum() * resolution) # MWh
    return PenDeficit

def penDC(x, S=None):
    S = Solution(x) if S is None else S
    TDC = Transmission(S) if scenario>=21 else np.zeros((intervals, len(DCloss))) # TDC: TDC(t, k), MW
    CDC = np.amax(abs(TDC), axis=0) * pow(10, -3) # CDC(k), MW to GW
    PenDC = max(0, CDC[6] - CDC6max) * pow(10, 3) # GW to MW
    return PenDC 

def lcoe(x, S=None):
    S = Solution(x) if S is None else S
    TDC = Transmission(S) if scenario>=21 else np.zeros((intervals, len(DCloss))) # TDC: TDC(t, k), MW
    CDC = np.amax(abs(TDC), axis=0) * pow(10, -3) # CDC(k), MW to GW

    Deficit, DeficitD = Reliability(S, flexible=np.zeros(intervals)) # Sj-EDE(t, j), MW
    Flexible = (Deficit + DeficitD / efficiencyD).sum() * resolution / years / (0.5 * (1 + efficiency)) # MWh p.a.
    Hydro = Flexible + GBaseload.sum() * resolution / years # Hydropower & biomass: MWh p.a.

    cost = factor * np.array([sum(S.CPV), sum(S.CWind), sum(S.CPHP), S.CPHS] + list(CDC) + [sum(S.CPV), sum(S.CWind), Hydro * pow(10, -6), -1, -1]) # $b p.a.
    if scenario<=17:
        cost[-1], cost[-2] = [0] * 2
    loss = np.sum(abs(TDC), axis=0) * DCloss
    loss = loss.sum() * pow(10, -9) * resolution / years # PWh p.a.
    LCOE = cost.sum() / abs(energy - loss)
    return LCOE

def F(x, S=None):
    """This is the objective function."""
    
    S = Solution(x) if S is None else S
    Deficit, DeficitD = Reliability(S, flexible=np.zeros(intervals)) # Sj-EDE(t, j), MW
    Flexible = (Deficit + DeficitD / efficiencyD).sum() * resolution / years / (0.5 * (1 + efficiency)) # MWh p.a.
    Hydro = Flexible + GBaseload.sum() * resolution / years # Hydropower & biomass: MWh p.a.
    PenHydro = max(0, Hydro - 20 * pow(10, 6)) # TWh p.a. to MWh p.a.
    
    Deficit, DeficitD = Reliability(S, flexible=np.ones(intervals) * CPeak.sum() * pow(10, 3)) # Sj-EDE(t, j), GW to MW
    PenDeficit = max(0, (Deficit + DeficitD / efficiencyD).sum() * resolution) # MWh
    
    TDC = Transmission(S) if scenario>=21 else np.zeros((intervals, len(DCloss))) # TDC: TDC(t, k), MW
    CDC = np.amax(abs(TDC), axis=0) * pow(10, -3) # CDC(k), MW to GW
    PenDC = max(0, CDC[6] - CDC6max) * pow(10, 3) # GW to MW

    cost = factor * np.array([sum(S.CPV), sum(S.CWind), sum(S.CPHP), S.CPHS] + list(CDC) + [sum(S.CPV), sum(S.CWind), Hydro * pow(10, -6), -1, -1]) # $b p.a.
    if scenario<=17:
        cost[-1], cost[-2] = [0] * 2
    loss = np.sum(abs(TDC), axis=0) * DCloss
    loss = loss.sum() * pow(10, -9) * resolution / years # PWh p.a.
    LCOE = cost.sum() / abs(energy - loss)
    
    if PenHydro+PenDeficit+PenDC > 0.1:
        return np.inf
    
    return LCOE


def F_v(x, callback=False):
    """This is the objective function."""
    
    if len(x.shape) > 1:
        with Pool(processes = min(x.shape[1], cpu_count())) as processPool:
            results = processPool.map(F, [xn for xn in x.T])
        results = np.array(results)
    else :
        results = np.array([F(x)])
        x = x.reshape(len(bounds), -1)
    
    if callback is True: 
        printout = np.concatenate((results.reshape(-1, 1), x.T), axis = 1)
        with open('Results/OpHist{}-{}.csv'.format(scenario, args.x), 'a', newline="") as csvfile:
            writer = csv.writer(csvfile)
            for row in printout:
                writer.writerow(row)
    
    return results

def F_d(x):
    func = F(x)

    with open('Results/dOpHist{}.csv'.format(scenario), 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([func] + list(x))

    return func