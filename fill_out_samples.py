# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 16:07:11 2024

@author: u6942852
"""
from scipy.optimize import differential_evolution
from argparse import ArgumentParser
import datetime as dt
import csv

from Setup import *

from Simulation import Reliability
from Network import Transmission
from multiprocessing import Pool, cpu_count

def normalised_dist_sq(x, x1):
    normalDif = (x-x1)/brange
    return np.square(normalDif).sum()

def distance(x, alts):
    print(alts)
    raise Exception
    return sum([maxDistSq-normalised_dist_sq(x, xn[1:]) for xn in alts])/alts.shape[0]

def distance_v(x, callback=False):
    alts = np.genfromtxt('Results/OpHist{}-{}.csv'.format(scenario, args.x), 
                        delimiter=',', dtype=float).reshape(-1, len(bounds)+1)
    alts = (alts.sum(axis=0)/alts.shape[1]).reshape(1,-1)
    """This is the objective function."""
    with Pool(processes = min(x.shape[1], cpu_count())) as processPool:
        results = processPool.starmap(distance, [(xn, alts) for xn in x.T])
        
        F_values = processPool.map(F, [xn for xn in x.T])
        
    results = np.array(results)
    F_values = np.array(F_values)
    
    if callback is True: 
        printout = np.concatenate((F_values.reshape(-1, 1), x.T), axis = 1)
        with open('Results/OpHist{}-{}.csv'.format(scenario, args.x), 'a', newline="") as csvfile:
            writer = csv.writer(csvfile)
            for row in printout:
                writer.writerow(row)
    
    return results

brange = np.array([ub-lb for lb, ub in bounds])
maxDistSq = len(bounds)

if __name__=='__main__':
    if args.x > 2: 
        x0 = np.ones(len(bounds))*(np.array(ub)/2)
        x0[args.x] = lb[args.x]
 
    elif args.x == 1: 
        x0 = np.genfromtxt('CostOptimisationResults/Optimisation_resultx{}.csv'.format(scenario), delimiter=',', dtype=float).reshape(-1,len(bounds))[0]
    elif args.x == 0:
        x0 = None

    starttime = dt.datetime.now()
    print("Optimisation starts at", starttime)

    workermap = Pool(processes=cpu_count()).map
    
    result = differential_evolution(
        func=distance_v,
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
        )

    # with open('Results/Optimisation_resultx{}-{}.csv'.format(scenario, args.x), 'a', newline="") as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(result.x)

    endtime = dt.datetime.now()
    print("Optimisation took", endtime - starttime)