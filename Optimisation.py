# To optimise the configurations of energy generation, storage and transmission assets
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

from scipy.optimize import differential_evolution
from argparse import ArgumentParser
import datetime as dt
import csv


parser = ArgumentParser()
parser.add_argument('-i', default=400, type=int, required=False, help='maxiter=4000, 400')
parser.add_argument('-p', default=1, type=int, required=False, help='popsize=2, 10')
parser.add_argument('-m', default=0.5, type=float, required=False, help='mutation=0.5')
parser.add_argument('-r', default=0.3, type=float, required=False, help='recombination=0.3')
parser.add_argument('-s', default=11, type=int, required=False, help='11, 12, 13, ...')
args = parser.parse_args()

scenario = args.s

from Input import *

def R(x, cost_constraint):
    """This is the new Resilience objective function""" 
    
    S = Solution(x) 
    
    if S.cost > cost_constraint:
        return 1000*(S.StormDeficit + S.Penalties)
    return S.StormDeficit + S.Penalties 


# if __name__=='__main__':
if 1==0:

    for cost_constraint in (110, 121, 132, 143):
        starttime = dt.datetime.now()
        print("Optimisation starts at", starttime)

        lb = [0.]  * pzones + [0.]   * wzones + contingency   + [0.]
        ub = [50.] * pzones + [50.]  * wzones + [50.] * nodes + [5000.]

        result = differential_evolution(func=R, bounds=list(zip(lb, ub)), args =[cost_constraint], tol=0,
                                        maxiter=args.i, popsize=args.p, mutation=args.m, recombination=args.r,
                                        disp=True, polish=False, updating='deferred', workers=-1)

        with open('Results/Optimisation_resultx{}.csv'.format(scenario), 'a', newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(result.x)

        with open('/Results/Otestx.csv'.format(scenario), 'a', newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([scenario,cost_constraint, result.fun, Solution(result.x).StormDeficit])
            writer.writerow(result.x)

        endtime = dt.datetime.now()
        print("Optimisation took", endtime - starttime)

        from Dispatch import Analysis
        Analysis(result.x)
    
