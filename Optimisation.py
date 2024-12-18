# To optimise the configurations of energy generation, storage and transmission assets
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

from scipy.optimize import differential_evolution
from datetime import datetime as dt

# from Costs import *
from Input import *
from Simulation import Reliability
from Network import Transmission

def F(x):
    S = Solution(x)
    return Objective(S, costs)

if __name__=='__main__':
    starttime = dt.now()
    print("Optimisation starts at", starttime)

    result = differential_evolution(
        func=F, 
        bounds=list(zip(lb, ub)), 
        tol=0,
        maxiter=args.i, 
        popsize=args.p, 
        mutation=(0.2, args.m),
        recombination=args.r,
        disp=True, 
        polish=False, 
        updating='deferred', 
        workers=-1,
        )
    try: 
        np.savetxt(f'Results/Optimisation_resultx{scenario}.csv', result.x.reshape(1,-1), fmt='%s', delimiter=',')
    except FileNotFoundError:
        import os
        os.mkdir('Results')
        np.savetxt(f'Results/Optimisation_resultx{scenario}.csv', result.x.reshape(1,-1), fmt='%s', delimiter=',')

    endtime = dt.now()
    print("Optimisation took", endtime - starttime)

    from Fill import Analysis
    Analysis(result.x)