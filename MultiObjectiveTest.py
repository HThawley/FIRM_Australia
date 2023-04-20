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
    
    return [S.StormDeficit, S.cost]


def is_pareto_efficient(costs, return_mask = True):
    """
    https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
    
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient

class Resilience(Problem):
    def __init__(self, pzones, wzones, contingency, nodes): 
        super().__init__(n_var = pzones + wzones + 2, 
                         n_obj = 2, 
                         n_ieq_constr = 0, 
                         xl =  np.array([0.] *pzones+[0.] * wzones + contingency + [0.]), 
                         xu =  np.array([50.]*pzones+[50.]* wzones + [50.]*nodes + [5000.]))
        
        
    def _evaluate(self, x, out, *args, **kwargs):
        out['F'] = objFunc(x, self.n_var)
        
    def _calc_pareto_set(self, solutionSpace, designSpace=None):
        
        mask = is_pareto_efficient(solutionSpace)
        if designSpace: 
            return solutionSpace[mask], designSpace[mask]
        else: 
            return solutionSpace[mask]
        

if __name__=='__main__':
    starttime = dt.datetime.now()
    print("Optimisation starts at", starttime)

    result = minimize(Resilience(pzones, wzones, contingency, nodes), 
                      NSGA2(pop_size = cpu_count()), 
                      ('n_gen', 1), 
                      seed = 1,
                      mutation=BitflipMutation(prob=0.5, prob_var=0.3),
                      crossover=SinglePointCrossover(prob=0.5),
                      verbose = True, 
                      save_history = True)
    
    with open('Results/MOOptimisation_result{}.csv'.format(scenario), 'a', newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(dt.datetime.now(), result.X)
    with open('Results/MOOptimisation_output{}.csv'.format(scenario), 'a', newline="") as csvfile:
         writer = csv.writer(csvfile)
         writer.writerow(dt.datetime.now(), result.F)
    
    solutionSpace = [e.opt.get("F") for e in result.history]
    designSpace = [e.opt.get("X") for e in result.history]
    
    solutions, designs = result._calculate_pareto_set(solutionSpace, 
                                                      designSpace)
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(211)
    plt.scatter(solutions[:,0], solutions[:,1], s=30, facecolors='none', edgecolors='b', label="Solutions")
    ax.set_title("Solutions")
    ax.set_xlabel("Fragility")
    ax.set_ylabel("Cost ($/MWh)")
        

    ax1 = fig.add_subplot(212)
    plt.scatter(result.F[:, 0], result.F[:, 1], s=30, facecolors='none', edgecolors='b', label="Solutions")
    # plt.plot(pf_a[:, 0], pf_a[:, 1], alpha=0.5, linewidth=2.0, color="red", label="Pareto-front")
    # plt.plot(pf_b[:, 0], pf_b[:, 1], alpha=0.5, linewidth=2.0, color="red")
    ax1.set_title("Objective Space")
    ax1.legend()
    plt.show()

    endtime = dt.datetime.now()
    print("Optimisation took", endtime - starttime)

    # from Dispatch import Analysis
    # Analysis(result.x)
    







