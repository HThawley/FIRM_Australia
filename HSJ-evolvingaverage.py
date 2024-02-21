# To optimise the configurations of energy generation, storage and transmission assets
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au


from Setup import *
import numpy as np
from scipy.optimize import differential_evolution, NonlinearConstraint
import csv
import datetime as dt
from scipy.optimize import minimize 


if args.c is not None:
    optimisedCost = np.genfromtxt('CostOptimisationResults/Costs.csv', delimiter=',', skip_header=1)
    optimisedCost = optimisedCost[optimisedCost[:,0] == scenario, 1][0]
    costConstraint = args.c*optimisedCost
else: 
    costConstraint = np.inf

def callback(xk, convergence=None):
    global ave_caps
    with open('Results/AltHist{}.csv'.format(scenario), 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([n, F(xk), objective(xk, ave_caps)] + list(xk))

def init_callbackfile(m):
    with open('Results/AltHist{}.csv'.format(scenario), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['alt', 'origfunc', 'HSJ func'] + ['dvar']*m)


def weighted_sum(x, ave_caps):
    weights = np.abs(1-x/ave_caps)   
    return (x*weights).sum()

def objective(x, ave_caps):
    func = weighted_sum(x, ave_caps)
    if F(x) > costConstraint:
        return func + 1e6
    return func

#%%

brange = np.array([ub-lb for lb, ub in bounds])
n = args.n

if __name__ == '__main__':
    x0 = np.genfromtxt('CostOptimisationResults/Optimisation_resultx{}.csv'.format(scenario), delimiter=',', dtype=float).reshape(-1,len(bounds))[0]

    if args.x == 1: 
        with open('Results/HSJ-ea_alternativesx{}.csv'.format(scenario), 'w', newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([0, None, F(x0)]+list(x0))
    assert args.x in (1,2)
    
    if args.his == 1: 
        init_callbackfile(len(bounds))
    
    for alt in range(n):
        starttime = dt.datetime.now()
        print(f"Beginning alternative {alt+1}/{n}.\nOptimisation starts at", starttime)
        
        ave_caps = np.genfromtxt('Results/HSJ-ea_alternativesx{}.csv'.format(scenario),
                                 delimiter=',', dtype=float).reshape(-1, len(bounds)+3)
        ave_caps = ave_caps[:,3:].sum(axis=0) / ave_caps.shape[0]
        
        # result = minimize(
        #     fun=objective,
        #     x0=x0,
        #     method='Nelder-Mead',
        #     bounds=bounds,
        #     options={
        #         'disp':True,
        #         'maxiter':args.i,
        #         'adaptive':False,
        #         },
        #     )
    
        result = differential_evolution(
            func=objective,
            x0=x0,
            args=(ave_caps,),
            bounds=bounds,
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
    
        with open('Results/HSJ-ea_alternativesx{}.csv'.format(scenario), 'a', newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([n, obj(result.x), F(result.x)] + list(result.x))
    
        endtime = dt.datetime.now()
        print("Optimisation took", endtime - starttime)

    
    
    

