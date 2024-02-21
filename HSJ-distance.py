# To optimise the configurations of energy generation, storage and transmission assets
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au


from Setup import *
import numpy as np
from scipy.optimize import differential_evolution, basinhopping
import csv
import datetime as dt
from multiprocessing import Pool, cpu_count

if args.c is not None:
    optimisedCost = np.genfromtxt('CostOptimisationResults/Costs.csv', delimiter=',', skip_header=1)
    optimisedCost = optimisedCost[optimisedCost[:,0] == scenario, 1][0]
    costConstraint = args.c*optimisedCost
else: 
    costConstraint = np.inf

def callback(xk, convergence=None):
    with open('Results/AltHist{}.csv'.format(scenario), 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([n, F(xk), objective(xk)] + list(xk))

def init_callbackfile(m):
    with open('Results/AltHist{}.csv'.format(scenario), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['alt', 'origfunc', 'HSJ func'] + ['dvar']*m)

def normalised_dist_sq(x, x1):
    normalDif = (x-x1)/brange
    return np.square(normalDif).sum()

def objective(x):
    alts = np.genfromtxt('Results/Optimisation_alternativesx{}.csv'.format(scenario), 
                         delimiter=',', dtype=float).reshape(-1, len(bounds)+2)
    func = sum([maxDistSq-normalised_dist_sq(x, xn[2:]) for xn in alts])/alts.shape[0]
    if F(x) > costConstraint:
        return 1e10/func 
    return func
#%%

# """ Custom step-function """
# class RandomDisplacementBounds(object):1e6
#     """random displacement with bounds:  see: https://stackoverflow.com/a/21967888/2320035
#         Modified! (dropped acceptance-rejection sampling for a more specialized approach)
#     """
#     def __init__(self, xmin, xmax, stepsize=0.5):
#         self.xmin = xmin
#         self.xmax = xmax
#         self.stepsize = stepsize

#     def __call__(self, x):
#         """take a random step but ensure the new position is within the bounds """
#         min_step = np.maximum(self.xmin - x, -self.stepsize)
#         max_step = np.minimum(self.xmax - x, self.stepsize)

#         random_step = np.random.uniform(low=min_step, high=max_step, size=x.shape)
#         xnew = x + random_step

#         return xnew

# def callback(x, f, accept):
#     with open('Results/AltHist{}.csv'.format(scenario), 'a', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow([f, accept, F(x)] + list(x))

# def init_callbackfile(m):
#     with open('Results/AltHist{}.csv'.format(scenario), 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(['func', 'accept', 'cost'] + ['dvar']*m)

# def basinHopWrapper(kwargs):
#     result = basinhopping(**kwargs)
#     return result

# def parallelbasinhopping(bargs, max_workers=-1):
#     if max_workers == -1:
#         max_workers = cpu_count()
    
#     with Pool(processes=max_workers) as processPool:
#         result = processPool.map(basinHopWrapper, [bargs for i in range(max_workers)])
    
#     result = [x for x in result]1e6
    
#     return result

brange = np.array([ub-lb for lb, ub in bounds])
maxDistSq = len(bounds)

if __name__ == '__main__':
    x0 = np.genfromtxt('CostOptimisationResults/Optimisation_resultx{}.csv'.format(scenario), delimiter=',', dtype=float).reshape(-1,len(bounds))[0]

    if args.x == 1: 
        with open('Results/Optimisation_alternativesx{}.csv'.format(scenario), 'w', newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([F(x0)]+list(x0))
    assert args.x in (1,2)
    
    n = args.n
    if args.his == 1: 
        init_callbackfile(len(bounds))
    
    for alt in range(n):
    
        starttime = dt.datetime.now()
        print(f"Beginning alternative {alt+1}/{n}.\nOptimisation starts at", starttime)
    
        # from scipy.optimize import minimize      
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
            bounds=bounds,
            tol=0,
            maxiter=args.i, 
            popsize=args.p, 
            mutation=args.m, 
            recombination=args.r,
            disp=bool(args.v), 
            polish=False, 
            updating='deferred', 
            workers=-1,
            callback=callback if args.his==1 else None,
            )
        
       
        # results = parallelbasinhopping(
        #     {
        #     'func':objective,
        #     'x0':x0,
        #     'niter':10,
        #     'T':0.4,
        #     'minimizer_kwargs':{
        #         'method':'Powell',
        #         'bounds':bounds,
        #         'options':{
        #             'disp':True,
        #             'maxiter':400, 
        #             # 'adaptive':True,
        #             },
        #     },
        #     'take_step':RandomDisplacementBounds(np.array(lb), np.array(ub)),
        #     'callback':callback,
        #     'disp':args.v,
        #     },
        #     -1)
    
        with open('Results/Optimisation_alternativesx{}.csv'.format(scenario), 'a', newline="") as csvfile:
            writer = csv.writer(csvfile)
            # for result in results:
            writer.writerow([objective(result.x), F(result.x)] + list(result.x))
    
        endtime = dt.datetime.now()
        print("Optimisation took", endtime - starttime)

    
    
    

