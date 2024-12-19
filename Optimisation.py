# To optimise the configurations of energy generation, storage and transmission assets
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

from scipy.optimize import differential_evolution
from datetime import datetime as dt
from multiprocessing import Pool
from psutil import cpu_count

ncpus = cpu_count(False)

# from Costs import *
from Input import *
from Simulation import Reliability
from Network import Transmission


def F(x):
    S = Solution(x)
    S._evaluate(costs)
    return S.LCOE + S.Penalties

def mpWrapper(x):
    with Pool(min(ncpus, x.shape[1])) as processPool:
        result = processPool.imap(F, [xn for xn in x.T], chunksize=x.shape[1]//ncpus+1)
        result = np.array([r for r in result])
        processPool.terminate()
    return result

if __name__=='__main__':
    starttime = dt.now()
    print("Optimisation starts at", starttime)

    result = differential_evolution(
        func=mpWrapper, 
        bounds=list(zip(lb, ub)), 
        tol=0,
        maxiter=args.i, 
        popsize=args.p, 
        mutation=(0.3, args.m),
        recombination=args.r,
        disp=True, 
        polish=False, 
        updating='deferred', 
        # workers=-1,
        vectorized=True,
        )
    try: 
        np.savetxt(f'Results/Optimisation_resultx{suffix}.csv', result.x.reshape(1,-1), fmt='%s', delimiter=',')
    except FileNotFoundError:
        import os
        os.mkdir('Results')
        np.savetxt(f'Results/Optimisation_resultx{suffix}.csv', result.x.reshape(1,-1), fmt='%s', delimiter=',')

    endtime = dt.now()
    print("Optimisation took", endtime - starttime)

    from Fill import Analysis
    Analysis(result.x)