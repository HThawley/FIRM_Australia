# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 16:25:06 2023

@author: hmtha
"""

import pymoo
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.core.callback import Callback
from pymoo.operators.crossover.pntx import SinglePointCrossover
import matplotlib.pyplot as plt

from scipy.optimize import differential_evolution
from argparse import ArgumentParser
import datetime as dt
import csv
from multiprocessing import Pool, cpu_count
import numpy as np


parser = ArgumentParser()
parser.add_argument('-i', default=400, type=int, required=False, help='maxiter=4000, 400')
parser.add_argument('-p', default=1, type=int, required=False, help='popsize=2, 10')
parser.add_argument('-m', default=0.5, type=float, required=False, help='mutation=0.5')
parser.add_argument('-r', default=0.3, type=float, required=False, help='recombination=0.3')
parser.add_argument('-s', default=12, type=int, required=False, help='11, 12, 13, ...')
args = parser.parse_args()

scenario = args.s

from Input import *

def objFunc(x, n_var):
    """This is the new Resilience objective function""" 
    return_values = [[]*len(x)]
    
    pool = Pool(processes=min(cpu_count(), len(x)))
    result = pool.map(R, x)
    # print(result)
    return result

def R(x):
    S = Solution(x)
    
    return [S.StormDeficit, S.cost + S.penalties]


class Resilience(Problem):
    def __init__(self, pzones, wzones, contingency, nodes): 
        super().__init__(n_var = pzones + wzones + 2, 
                         n_obj = 2, 
                         n_ieq_constr = 0, 
                         xl =  np.array([0.] *pzones+[0.] * wzones + contingency + [0.]), 
                         xu =  np.array([50.]*pzones+[50.]* wzones + [50.]*nodes + [5000.]))
        
        
    def _evaluate(self, x, out, *args, **kwargs):
        out['F'] = objFunc(x, self.n_var)
        

if __name__=='__main__':
    starttime = dt.datetime.now()
    print("Optimisation starts at", starttime)

<<<<<<< HEAD
    try: 
        result = minimize(Resilience(pzones, wzones, contingency, nodes), 
                          NSGA2(pop_size = cpu_count()), 
                          ('n_gen', 1), 
                          # seed = 1,
                          mutation=BitflipMutation(prob=0.5, prob_var=0.3),
                          crossover=SinglePointCrossover(prob=0.5),
                          verbose = True
                          # , save_history = True
                          )
    except KeyboardInterrupt:
        pass
=======
    result = minimize(Resilience(pzones, wzones, contingency, nodes), 
                      NSGA2(pop_size = cpu_count()), 
                      ('time', "19:00:00"), 
                      seed = 1,
                      mutation=BitflipMutation(prob=0.5, prob_var=0.3),
                      crossover=SinglePointCrossover(prob=0.5),
                      verbose = True, 
                      save_history = True)
>>>>>>> 508a501a48c0648ce1e1ae7481a58ecc44a2fd13
    
    # with open('Results/MOOptimisation_result{}.csv'.format(scenario), 'a', newline="") as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(dt.datetime.now(), result.X)
    # with open('Results/MOOptimisation_output{}.csv'.format(scenario), 'a', newline="") as csvfile:
    #      writer = csv.writer(csvfile)
    #      writer.writerow(dt.datetime.now(), result.F)
         
    with open('/media/fileshare/FIRM_Aus_Resilience2/testX.csv', 'a', newline="") as csvfile:
        writer = csv.writer(csvfile)
<<<<<<< HEAD
        writer.writerow(result.X[:,0])
        writer.writerow(result.X[:,1])
        writer.writerow(result.X[:,2])
        writer.writerow(result.X[:,3])
    with open('/media/fileshare/FIRM_Aus_Resilience2/testF.csv', 'a', newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(result.F[:,0])
        writer.writerow(result.F[:,1])

    print(result.F)
    endtime = dt.datetime.now()
    print("Optimisation took", endtime - starttime)

=======
        writer.writerow(result.X)
    with open('Results/MOOptimisation_output{}.csv'.format(scenario), 'a', newline="") as csvfile:
         writer = csv.writer(csvfile)
         writer.writerow(result.F)
    
    solutionSpace = [e.opt.get("F") for e in result.history]
    designSpace = [e.opt.get("X") for e in result.history]
    
    solutions, designs = result._calculate_pareto_set(solutionSpace, 
                                                      designSpace)
    
>>>>>>> 508a501a48c0648ce1e1ae7481a58ecc44a2fd13

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    plt.scatter(result.F[:, 0], result.F[:, 1], s=30, facecolors='none', edgecolors='b', label="Solutions")
    ax.set_title("Solutions")
    ax.set_xlabel("Fragility")
    ax.set_ylabel("Cost ($/MWh)")
        


    from Dispatch import Analysis
    for r in result.X:
        Analysis(r)
    
    plt.show()






