# To optimise the configurations of energy generation, storage and transmission assets
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

from scipy.optimize import differential_evolution
from argparse import ArgumentParser
import datetime as dt
import csv
from multiprocessing import Pool, cpu_count

parser = ArgumentParser()
parser.add_argument('-i', default=400, type=int, required=False, help='maxiter=4000, 400')
parser.add_argument('-p', default=1, type=int, required=False, help='popsize=2, 10')
parser.add_argument('-m', default=0.5, type=float, required=False, help='mutation=0.5')
parser.add_argument('-r', default=0.3, type=float, required=False, help='recombination=0.3')
parser.add_argument('-s', default=11, type=int, required=False, help='11, 12, 13, ...')
parser.add_argument('-cb', default=0, type=int, required=False, help='Callback: 0-None, 1-population elites, 2-everything')
parser.add_argument('-vp', default=50, type=int, required=False, help='Maximum number of vectors to send to objective')
parser.add_argument('-w', default=-1, type=int, required=False, help='Maximum number of cores to parallelise over')
args = parser.parse_args()

scenario = args.s

from Input import *
from Simulation import Reliability
from Network import Transmission

def F(x, n=0):
    """This is the objective function."""
    S = Solution(x, vectorize=True)

    Deficit, DeficitD = Reliability(S, flexible=np.zeros((intervals, 1))) # Sj-EDE(t, j), MW
    Flexible = (Deficit + DeficitD / efficiencyD).sum(axis=0) * resolution / years / (0.5 * (1 + efficiency)) # MWh p.a.
    Hydro = Flexible + GBaseload.sum() * resolution / years # Hydropower & biomass: MWh p.a.
    PenHydro = np.maximum(0, Hydro - 20 * pow(10, 6)) # TWh p.a. to MWh p.a.

    Deficit, DeficitD = Reliability(S, flexible=np.ones((intervals, 1)) * CPeak.sum() * pow(10, 3)) # Sj-EDE(t, j), GW to MW
    PenDeficit = np.maximum(0, (Deficit + DeficitD / efficiencyD).sum(axis=0) * resolution) # MWh

    TDC = Transmission(S) if scenario>=21 else np.zeros((intervals, len(DCloss))) # TDC: TDC(t, k), MW
    CDC = np.amax(np.abs(TDC), axis=0) * pow(10, -3) # CDC(k), MW to GW
    PenDC = np.maximum(0, CDC[6,:] - CDC6max) * pow(10, 3) # GW to MW

    cost = factor.reshape(-1,1) * np.array(
        [S.CPV.sum(axis=0), S.CWind.sum(axis=0), S.CPHP.sum(axis=0), S.CPHS] + list(CDC) + 
        [S.CPV.sum(axis=0), S.CWind.sum(axis=0), Hydro * pow(10, -6)] + 2*[np.repeat(-1, x.shape[1])]) # $b p.a.
    if scenario<=17:
        cost[-1,:], cost[-2,:] = [np.zeros(x.shape[1])] * 2
    cost = cost.sum(axis=0)
    loss = np.abs(TDC).sum(axis=(0)) * DCloss.reshape(-1,1)
    loss = loss.sum(axis=0) * pow(10, -9) * resolution / years # PWh p.a.
    LCOE = cost / np.abs(energy - loss)

    Func = LCOE + PenHydro + PenDeficit + PenDC

    return n, Func

def F_wrapper(x, callback=False):
    npop = x.shape
    if npop > 1:
        with Pool(processes=min(x.shape[1], cpu_count())) as processPool:
            r = range(npop//vp) if npop%vp == 0 else range(npop//vp + 1)
            results = processPool.starmap(F, [(x[:, n*vp: min(n+1*vp, npop-1)], n) for n in r])
        results = np.array(results, dtype=object)
        results = np.concatenate(results[results[:,0].argsort(),1])
    
    if callback is True: 
        printout = np.concatenate((results.reshape(-1, 1), x.T), axis = 1)
        with open('Results/History{}.csv'.format(scenario), 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(printout)
    
    return results

def callback(xk, convergence=None):
    with open('Results/History{}.csv'.format(scenario), 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([F(xk)] + list(xk))
        
def init_callback():
    with open('Results/History{}.csv'.format(scenario), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        
if __name__=='__main__':
    starttime = dt.datetime.now()
    print("Optimisation starts at", starttime)

    lb = [0.]  * pzones + [0.]   * wzones + contingency   + [0.]
    ub = [50.] * pzones + [50.]  * wzones + [50.] * nodes + [5000.]

    result = differential_evolution(
        func=F_wrapper, 
        args=(args.cb==2,),
        bounds=list(zip(lb, ub)), 
        tol=0,
        maxiter=args.i, 
        popsize=args.p, 
        mutation=args.m, 
        recombination=args.r,
        disp=True, 
        polish=False, 
        updating='deferred', 
        vectorize=True,
        callback=callback if args.cb == 1 else None
        )

    with open('Results/Optimisation_resultx{}.csv'.format(scenario), 'a', newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(result.x)

    endtime = dt.datetime.now()
    print("Optimisation took", endtime - starttime)

    from Dispatch import Analysis
    Analysis(result.x)