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
parser.add_argument('-i', default=400, type=int, required=False, help='maxiter=4000, 400')
parser.add_argument('-p', default=2, type=int, required=False, help='popsize=2, 10')
parser.add_argument('-m', default=0.5, type=float, required=False, help='mutation=0.5')
parser.add_argument('-r', default=0.3, type=float, required=False, help='recombination=0.3')
parser.add_argument('-s', default=11, type=int, required=False, help='11, 12, 13, ...')
parser.add_argument('-c', default=1.5, type=float, required=False, help='cost constraint as multiplier of optimised cost')
parser.add_argument('-z', default='all', type=str, required=False, help='\'all\' or space/comma seperated list of zones (as int)')
parser.add_argument('-relative', default=True, type=bool, required=False,help='whether to use relative probability or pure highWindFrac')
args = parser.parse_args()

scenario = args.s
costConstraintFactor = args.c
relative = args.relative

def readPrintedArray(txt):      
    txt = sub(r"(?<!\[)\s+(?!\])", r",", txt)
    return np.array(literal_eval(txt), dtype =int)

stormZone = 'all' if args.z == 'all' else readPrintedArray(args.z)

from Input import *

stormZone = np.arange(wzones) if stormZone == 'all' else stormZone

def R(x):
    """This is the new Resilience objective function""" 

    S = Solution(x, stormZone) 
    
    StormDeficit, penalties, cost = S.StormDeficit, S.penalties, S.cost
    
    if penalties > 0: penalties = penalties*pow(10,6)
    
    func = StormDeficit + penalties + cost
    
    if cost > costConstraint: func = func*pow(10,6)
    
    return func

if __name__=='__main__':
    
    starttime = dt.datetime.now()
    print("Optimisation starts at", starttime)

    lb = [0.]  * pzones + [0.]   * wzones + contingency   + [0.] 
    ub = [50.] * pzones + [50.]  * wzones + [50.] * nodes + [5000.] 

    result = differential_evolution(func=R, bounds=list(zip(lb, ub)), tol=0, 
                                    maxiter=args.i, popsize=args.p, mutation=args.m, recombination=args.r,
                                    disp=True, polish=False, updating='deferred', workers=-1, x0 = x0)

    with open('Results/Optimisation_resultx{}-{}-{}.csv'.format(scenario, stormZone, relative), 'a', newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(list(result.x))
        
    S = Solution(result.x, stormZone)
    with open('Results/Otestx{}-{}-{}.csv'.format(scenario, stormZone, relative), 'a', newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([cost_constraint, S.cost, S.penalties, S.StormDeficit, result.fun, result.x])
    del S, csvfile
    
    endtime = dt.datetime.now()
    print("Optimisation took", endtime - starttime)

    from Dispatch import Analysis
    Analysis(result.x, stormZone)
    
    # from Statistics import Information
    # Information(result.x, np.zeros((175344,)))
