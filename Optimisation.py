# To optimise the configurations of energy generation, storage and transmission assets
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

from scipy.optimize import differential_evolution
from argparse import ArgumentParser
import datetime as dt
import csv
import numpy as np 
from re import sub
from ast import literal_eval    

parser = ArgumentParser()
parser.add_argument('-i', default=500,  type=int,   required=False, help='maxiter=4000, 400')
parser.add_argument('-p', default=4,    type=int,   required=False, help='popsize=2, 10')
parser.add_argument('-m', default=0.5,  type=float, required=False, help='mutation=0.5')
parser.add_argument('-r', default=0.3,  type=float, required=False, help='recombination=0.3')
parser.add_argument('-s', default=15,   type=int,   required=False, help='11, 12, 13, ...')
parser.add_argument('-c', default=1.5,  type=float, required=False, help='cost constraint as multiplier of optimised cost')
parser.add_argument('-z', default='[24]',type=str,   required=False, help="'None','All', or space/comma seperated, [] bracketed, list of zones (as int)")
parser.add_argument('-y', default=0,    type=int,   required=False, help='boolean whether to use relative probability or pure highWindFrac')
parser.add_argument('-v', default=1,    type=int,   required=False, help='boolean whether to print out optimisation at each step')
parser.add_argument('-n', default=25,   type=int,   required=False, help='1-in-N-year to consider, -2 uses value in windFragility.csv, -1 is no storm.')
parser.add_argument('-x', default=1,    type=int,   required=False, help="What first approx to use. 0-none, 1-Bin's results, 2-where it last ended. Note: if it can't find, it will try the next.")
args = parser.parse_args()

scenario = args.s
n_year = args.n
costConstraintFactor = args.c
relative = bool(args.y)
x0mode = args.x


def readPrintedArray(txt):      
    txt = sub(r"(?<!\[)\s+(?!\])", r",", txt)
    return np.array(literal_eval(txt), dtype =int)

try: stormZone = readPrintedArray(args.z)
except (TypeError, ValueError): stormZone = 'All' if args.z.lower()=='all' else 'None' if args.z.lower()=='none' else args.z

from Input import *

def R(x):
    """This is the new Resilience objective function""" 

    S = Solution(x) 
    
    StormDeficit, penalties, cost = S.StormDeficit, S.penalties, S.cost
    
    if penalties > 0: penalties = penalties*pow(10,6)
    
    func = StormDeficit + penalties + cost
    
    if cost > costConstraint: func = func*pow(10,6)
    
    return func

def callback(x_k, convergence):
    history.append(R(x_k))


if __name__=='__main__':
    
    starttime = dt.datetime.now()
    print("Optimisation starts at", starttime)

    lb = [0.]  * pzones + [0.]   * wzones + contingency   + [0.] 
    ub = [50.] * pzones + [50.]  * wzones + [50.] * nodes + [5000.] 

    history = []

    result = differential_evolution(func=R, bounds=list(zip(lb, ub)), tol=0, x0 = x0, 
                                    maxiter=args.i, popsize=args.p, mutation=args.m, recombination=args.r,
                                    disp=bool(args.v), polish=False, updating='deferred', workers=-1, 
                                    callback = callback)

    with open('Results/Optimisation_resultx{}-{}-{}.csv'.format(scenario, stormZone, n_year), 'w', newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(list(result.x))
    del csvfile
    
    history = np.array(history)
    if x0mode == 2: 
        try: history = np.append(np.genfromtxt('Results/OptimisationHistory{}-{}-{}.csv'.format(scenario, stormZone, n_year), delimiter=','), history)
        except FileNotFoundError: pass

    np.savetxt('Results/OptimisationHistory{}-{}-{}.csv'.format(scenario, stormZone, n_year), history, delimiter=',')


    endtime = dt.datetime.now()
    print("Optimisation took", endtime - starttime)

    from Dispatch import Analysis
    Analysis(result.x)

