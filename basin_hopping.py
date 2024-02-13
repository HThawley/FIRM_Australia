# To optimise the configurations of energy generation, storage and transmission assets
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

from scipy.optimize import basinhopping
from argparse import ArgumentParser
import datetime as dt
import csv

from Setup import *

from Simulation import Reliability
from Network import Transmission


""" Custom step-function """
class RandomDisplacementBounds(object):
    """random displacement with bounds:  see: https://stackoverflow.com/a/21967888/2320035
        Modified! (dropped acceptance-rejection sampling for a more specialized approach)
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
    with open(cbfile, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([f, accept] + list(x))

def init_callbackfile(n):
    with open(cbfile, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['func', 'accept'] + ['dvar']*n)

if __name__=='__main__':
    starttime = dt.datetime.now()
    print("Optimisation starts at", starttime)

    if args.x == 2: 
        x0 = np.genfromtxt('Results/Optimisation_resultx{}.csv'.format(scenario), delimiter=',', dtype=float)
    elif args.x == 1:
        x0 = np.genfromtxt('costOptimisationResults/Optimisation_resultx{}-None.csv'.format(scenario), delimiter=',', dtype=float)
    else:
        x0 = np.random.rand(len(ub)) * np.array(ub)
        
    if args.his == 1: 
        cbfile = 'Results/bhopOptimHistory{}.csv'.format(scenario) 
        init_callbackfile(len(ub))
    else:
        cbfile = None
    
    bounded_step = RandomDisplacementBounds(np.array(lb), np.array(ub))
        
    result = basinhopping(
        func=F, 
        x0=x0,
        minimizer_kwargs={
            "method":"Powell",
            "bounds":bounds,
            "options":{"maxiter":1000*len(bounds)},
            },
        niter=3, 
        T=0.4,
        disp=bool(args.v), 
        take_step=bounded_step,
        callback = callback,
        )

    with open('Results/Optimisation_resultx{}.csv'.format(scenario), 'a', newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(result.x)

    endtime = dt.datetime.now()
    print("Optimisation took", endtime - starttime)

    # from Dispatch import Analysis
    # Analysis(result.x)