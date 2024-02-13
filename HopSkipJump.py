# To optimise the configurations of energy generation, storage and transmission assets
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au


from Setup import *
from scipy.optimize import differential_evolution
import csv
import datetime as dt

if args.c is not None:
    optimisedCost = np.genfromtxt('CostOptimisationResults/Costs.csv', delimiter=',', skip_header=1)
    optimisedCost = optimisedCost[optimisedCost[:,0] == scenario, 1][0]
    costConstraint = args.c*optimisedCost
else: 
    costConstraint = np.inf

def callback(xk, convergence=None):
    with open('Results/AltHist{}-{}.csv'.format(scenario, n), 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([F(xk), objective(xk)] + list(xk))

def init_callbackfile(n, m):
    with open('Results/AltHist{}-{}.csv'.format(scenario, n), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['origfunc', 'HSJ func'] + ['dvar']*m)

def solutionDif(x, x1):
    return 1-np.absolute(x-x1)/maxDif

def objective(x):
    alts = np.genfromtxt('Results/Optimisation_alternativesx{}.csv'.format(scenario)
                                 , delimiter=',', dtype=float).reshape(-1, len(bounds)+1)
    func = np.array([solutionDif(x, xn[1:]) for xn in alts]).sum()/(alts.shape[0]*alts.shape[1])
        
    if F(x) > costConstraint:
        func = func*10e6
        
    return func
#%%

maxDif = np.array([ub-lb for lb, ub in bounds])

if __name__ == '__main__':
        
    x0 = np.genfromtxt('Results/Optimisation_resultx{}.csv'.format(scenario), delimiter=',', dtype=float)[0]

    if args.x >= 2:
        alternatives = np.genfromtxt('Results/Optimisation_alternativesx{}.csv'.format(scenario), delimiter=',', dtype=float)
    if args.x == 1: 
        with open('Results/Optimisation_alternativesx{}.csv'.format(scenario), 'w', newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([F(x0)]+list(x0))
        alternatives = np.genfromtxt('Results/Optimisation_alternativesx{}.csv'.format(scenario), delimiter=',', dtype=float)
    assert args.x in (1,2)
    
    n = args.n
    if args.his == 1: 
        init_callbackfile(n, len(bounds))
    
    for alt in range(n):
    
        starttime = dt.datetime.now()
        print(f"Beginning alternative {alt+1}/{n}.\nOptimisation starts at", starttime)
    
        result = differential_evolution(
            func=objective,
            x0=x0,
            bounds=list(zip(lb, ub)),
            tol=0,
            maxiter=args.i, 
            popsize=args.p, 
            mutation=args.m, 
            recombination=args.r,
            disp=True, 
            polish=False, 
            updating='deferred', 
            workers=-1,
            callback=callback if args.his==1 else None,
            )
    
        with open('Results/Optimisation_alternativesx{}.csv'.format(scenario), 'a', newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([F(x)] + list(result.x))
    
        endtime = dt.datetime.now()
        print("Optimisation took", endtime - starttime)

    
    
    

