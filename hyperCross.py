# To optimise the configurations of energy generation, storage and transmission assets
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au


from Setup import *
from Optimisation_local import local_sampling
import pandas as pd
import csv
from datetime import datetime


if args.c is not None:
    OptimisedCost = pd.read_csv('CostOptimisationResults/Costs.csv', index_col='Scenario').loc[scenario, 'LCOE']
    costConstraint = args.c*OptimisedCost
else: 
    costConstraint = np.inf

def HC_weights(pv=None, wind=None, storP=None, storE=None):
    for source in (pv, wind, storP, storE):
        assert source in (None, 'high', 'low')
    
    fac = {None:0, 'high':-0.5, 'low':0.5}
    CPV, CWind, CPHP, CPHS = (fac[source] for source in (pv, wind, storP, storE))
    return CPV, CWind, CPHP, CPHS


def writerow_callbackfile(xk):
    with open(cbfile, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([F(xk), func(xk)] + list(xk))


def init_callbackfile(n):
    with open(cbfile, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['cost', 'objective'] + ['dvar']*n)

def scipy_DE():
    global cbfile
    if args.his == 1: 
        cbfile = 'Results/HCminimOptimHistory{}.csv'.format(scenario) 
        init_callbackfile(len(ub))
    else:
        cbfile = None
    
    from scipy.optimize import differential_evolution
    
    start = datetime.now()
    print(f"Optimisation starts: {start}\n{'-'*40}")
    
    result = differential_evolution(
        objective, 
        x0=x0, 
        bounds = list(zip(lb, ub)),
        callback=None if cbfile is None else writerow_callbackfile,
        tol=0, 
        maxiter=args.i, 
        popsize=args.p, 
        mutation=args.m, 
        recombination=args.r,
        disp=bool(args.v), 
        polish=False, 
        updating='deferred', 
        workers=-1, 
        )
        
    end = datetime.now()
    print(f"Optimisation ends: {end}\nOptimisation took: {end-start}\n{'-'*40}")
    return result

def scipy_minimize(method='Powell'):
    global cbfile
    if args.his == 1: 
        cbfile = 'Results/HCminimOptimHistory{}.csv'.format(scenario) 
        init_callbackfile(len(ub))
    else:
        cbfile = None
    
    from scipy.optimize import minimize
    
    start = datetime.now()
    print(f"Optimisation starts: {start}\n{'-'*40}")
    
    result = minimize(
        fun=objective, 
        x0=x0, 
        method = method,
        bounds = list(zip(lb, ub)),
        callback=None if cbfile is None else writerow_callbackfile,
        )
    
    end = datetime.now()
    print(f"Optimisation ends: {end}\nOptimisation took: {end-start}\n{'-'*40}")
    return result

def local_optimiser():
    global cbfile
    if args.his == 1: 
        cbfile = 'Results/HClocalOptimHistory{}.csv'.format(scenario) 
    else:
        cbfile = None
    start = datetime.now()
    print(f"Optimisation starts: {start}\n{'-'*40}")
    ws, termination = local_sampling(
        func=objective,
        x0=x0,        
        bounds=list(zip(lb, ub)), 
        maxiter=args.i,
        disp=bool(args.v),
        incs=[(10**n) for n in range(0, -6, -1)],
        callback=cbfile,
        convex=None,
        )
    end = datetime.now()
    print(f"Optimisation ends: {end}\nOptimisation took: {end-start}\n{'-'*40}")
    return ws, termination

CPV, CWind, CPHP, CPHS = HC_weights('low', None, None, None)

def objective(x):
    S = Solution(x)
    obj = F(x) + CPV*sum(S.CPV) + CWind*sum(S.CWind) + CPHP*sum(S.CPHP) + CPHS*S.CPHS
    if F(x, S) > costConstraint:
        obj = abs(obj)*10e6
    return obj
#%%
if __name__ == '__main__':
        
    if args.x == 2: 
        x0 = np.genfromtxt('Results/Optimisation_resultx{}.csv'.format(scenario), delimiter=',', dtype=float)
    elif args.x == 1:
        x0 = np.genfromtxt('costOptimisationResults/Optimisation_resultx{}-None.csv'.format(scenario), delimiter=',', dtype=float)
    else:
        x0 = np.random.rand(len(ub)) * np.array(ub)
    
    # result = scipy_minimize()
    result = scipy_DE()
    # ws, termination = local_optimizer()
    
    
#%%


#%%
def quartile_sums(arr):
    """
    Generates an array containing the sums of each quartile of a given array
        the lower bound is included in each quartile sum
        the upper bound is excluded in each quartile sum, except the 4th quartile
    """
    q_sums = np.array(
        [(arr[(arr >= i[0]) * (arr < i[1])]).sum() 
         for i in np.lib.stride_tricks.sliding_window_view(
                 np.quantile(arr, np.arange(0,1.25,0.25)), 2
                 )
         ]
        ) 
    
    q_sums[-1] += x.max()
    return q_sums
