# To optimise the configurations of energy generation, storage and transmission assets
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

from scipy.optimize import basinhopping
import datetime as dt
import csv
import numpy as np
import multiprocessing as mp

from Setup import *

class RandomDisplacementBounds(object):
    """
    random displacement with bounds:  see: https://stackoverflow.com/a/21967888/2320035
    """
    def __init__(self, xmin, xmax, stepsize=0.5):
        self.xmin = xmin
        self.xmax = xmax
        self.stepsize = stepsize

    def __call__(self, x):
        """take a random step but ensure the new position is within the bounds """
        min_step = np.maximum(self.xmin - x, -self.stepsize)
        max_step = np.minimum(self.xmax - x, self.stepsize)
        
        random_step = np.random.uniform(low=min_step, high=max_step, size=x.shape)
        xnew = x + random_step
        
        return xnew
    
def callback(x, f, accept):
    with open('Results/bhopOptimHistory{}.csv'.format(scenario), 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([f, accept] + list(x))

def init_callbackfile(n):
    with open('Results/bhopOptimHistory{}.csv'.format(scenario), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['func', 'accept'] + ['dvar']*n)
        
def parallelbasinhopping(bargs, max_workers=-1):
    if max_workers == -1:
        max_workers = mp.cpu_count()
    
    with mp.Pool(processes=max_workers) as processPool:
        arglist = [bargs for i in range(max_workers)]
        result = processPool.starmap(basinhopping, arglist)
    
    result = [x for x in result]
    
    return result

if __name__=='__main__':
    starttime = dt.datetime.now()
    print("Optimisation starts at", starttime)
    if args.x == 2: 
        x0 = np.genfromtxt('Results/Optimisation_resultx{}.csv'.format(scenario), delimiter=',', dtype=float)
    elif args.x == 1:
        x0 = np.genfromtxt('CostOptimisationResults/Optimisation_resultx{}.csv'.format(scenario), delimiter=',', dtype=float)
    else:
        x0 = np.random.rand(len(bounds)) * np.array(ub)
    
    if args.his == 1: 
        init_callbackfile(len(bounds))
    
    bounded_step = RandomDisplacementBounds(np.array(lb), np.array(ub))
    
    # result = parallelbasinhopping(
    #     f=F, 
    #     x0=x0,
    #     bkwargs={
    #         'minimizer_kwargs':{"method":"L-BFGS-B"},
    #         'niter':1,
    #         'T':0.4,
    #         'disp':bool(args.v),
    #         'take_step':bounded_step,
    #         'callback':callback,
    #         },
    #     max_workers=-1
    #     )
    
    result = parallelbasinhopping(
        (F,     # func
         x0,    # x0
         1,     # niter
         0.4,   # T
         0.5,   # stepsize
         {"method":"Powell",
          "bounds":bounds,
          "options":{"maxiter":1000*len(bounds)}
          },            # minimizer_kwargs
         bounded_step,  # take_step
         None,          # accept_test
         callback,      # callback
         50,            # interval
         True,          # disp
            ),
        2)
    
    with open('Results/Optimisation_resultx{}.csv'.format(scenario), 'a', newline="") as csvfile:
        writer = csv.writer(csvfile)
        for res in result: 
            writer.writerow(res.x)
    
    endtime = dt.datetime.now()
    print("Optimisation took", endtime - starttime)
