# To optimise the configurations of energy generation, storage and transmission assets
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

from scipy.optimize import differential_evolution, NonlinearConstraint
from argparse import ArgumentParser
import datetime as dt
import csv

from Setup import *

from Simulation import Reliability
from Network import Transmission
from multiprocessing import Pool, cpu_count


# def F_callback(x, S=None):
#     """This is the objective function."""
    

#     S = Solution(x) if S is None else S
#     Deficit, DeficitD = Reliability(S, flexible=np.zeros(intervals)) # Sj-EDE(t, j), MW
#     Flexible = (Deficit + DeficitD / efficiencyD).sum() * resolution / years / (0.5 * (1 + efficiency)) # MWh p.a.
#     Hydro = Flexible + GBaseload.sum() * resolution / years # Hydropower & biomass: MWh p.a.
#     PenHydro = max(0, Hydro - 20 * pow(10, 6)) # TWh p.a. to MWh p.a.
    
#     Deficit, DeficitD = Reliability(S, flexible=np.ones(intervals) * CPeak.sum() * pow(10, 3)) # Sj-EDE(t, j), GW to MW
#     PenDeficit = max(0, (Deficit + DeficitD / efficiencyD).sum() * resolution) # MWh
    
#     TDC = Transmission(S) if scenario>=21 else np.zeros((intervals, len(DCloss))) # TDC: TDC(t, k), MW
#     CDC = np.amax(abs(TDC), axis=0) * pow(10, -3) # CDC(k), MW to GW
#     PenDC = max(0, CDC[6] - CDC6max) * pow(10, 3) # GW to MW

#     cost = factor * np.array([sum(S.CPV), sum(S.CWind), sum(S.CPHP), S.CPHS] + list(CDC) + [sum(S.CPV), sum(S.CWind), Hydro * pow(10, -6), -1, -1]) # $b p.a.
#     if scenario<=17:
#         cost[-1], cost[-2] = [0] * 2
#     loss = np.sum(abs(TDC), axis=0) * DCloss
#     loss = loss.sum() * pow(10, -9) * resolution / years # PWh p.a.
#     LCOE = cost.sum() / abs(energy - loss)
    
#     penalties = PenHydro + PenDeficit + PenDC
#     with open('Results\OpHist{}.csv'.format(scenario), 'a', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow([LCOE, penalties, LCOE + penalties] + list(x))
        
#     return LCOE + penalties

if __name__=='__main__':
    if args.x == 1: 
        x0 = np.genfromtxt('CostOptimisationResults/Optimisation_resultx{}.csv'.format(scenario), delimiter=',', dtype=float).reshape(-1,len(bounds))[0]
    elif args.x == 0:
        x0 = None

    starttime = dt.datetime.now()
    print("Optimisation starts at", starttime)

    workermap = Pool(processes=cpu_count()).map
    
    result = differential_evolution(
        func=F_v,
        x0=x0, 
        args=(True,),
        bounds=list(zip(lb, ub)),
        tol=0,
        maxiter=args.i, 
        popsize=cpu_count(), 
        mutation=args.m, 
        recombination=args.r,
        disp=bool(args.v), 
        polish=False, 
        updating='deferred', 
        # workers=-1,
        vectorized=True,
        # constraints=[
        #     NonlinearConstraint(penalties, -np.inf, 0),
        #     # NonlinearConstraint(penHydro, -np.inf, 0),
        #     # NonlinearConstraint(penDeficit, -np.inf, 0),
        #     # NonlinearConstraint(penDC, -np.inf, 0),
        #     ]
        )

    with open('Results/Optimisation_resultx{}.csv'.format(scenario), 'a', newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(result.x)

    endtime = dt.datetime.now()
    print("Optimisation took", endtime - starttime)

    # from Dispatch import Analysis
    # Analysis(result.x)