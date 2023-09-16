# Step-by-step analysis to decide the dispatch of flexible energy resources
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

from Input import *
from Simulation import Reliability
from CoSimulation import Resilience

import datetime as dt
from multiprocessing import Pool, cpu_count

def Flexible(instance, resilience=False):
    """Energy source of high flexibility"""
    year, x = instance
    print(year, end=', ')

    S = Solution(x)

    startidx = int((24 / resolution) * (dt.datetime(year, 1, 1) - dt.datetime(firstyear, 1, 1)).days)
    endidx = int((24 / resolution) * (dt.datetime(year+1, 1, 1) - dt.datetime(firstyear, 1, 1)).days)

    Fcapacity = CPeak.sum() * pow(10, 3) # GW to MW
    flexible = Fcapacity * np.ones(endidx - startidx)
    rflexible = Fcapacity * np.ones(endidx - startidx)

    for i in range(0, endidx - startidx, timestep):
        flexible[i: i+timestep] = 0
        rflexible[i: i+timestep] = 0
        Deficit, DeficitD, RDeficit, RDeficitD = Resilience(S, flexible=rflexible, start=startidx, end=endidx, output='deficits') # Sj-EDE(t, j), MW
        Deficit, DeficitD = Reliability(S, flexible=flexible, start=startidx, end=endidx, output=True) # Sj-EDE(t, j), MW
        if (Deficit + DeficitD).sum() * resolution > 0.1:
            flexible[i: i+timestep] = Fcapacity
        if (RDeficit + RDeficitD).sum() * resolution > 0.1:
            rflexible[i: i+timestep] = Fcapacity
            
    flexible = np.clip(flexible - S.Spillage, 0, None)
    rflexible = np.clip(rflexible - S.RSpillage, 0, None)

    if resilience == False: return flexible
    else: return flexible, rflexible


def Analysis(x):
    """Dispatch.Analysis(result.x)"""

    starttime = dt.datetime.now()
    print('Dispatch starts at', starttime)
    print('Dispatch works on: ', end='')
    # Multiprocessing
    pool = Pool(processes=min(cpu_count(), finalyear - firstyear + 1))
    instances = map(lambda y: [y] + [x], range(firstyear, finalyear + 1))
 
    Dispresult = pool.map(Flexible, instances)
    pool.terminate()
    
    Flex, RFlex = tuple(zip(*Dispresult))
    Flex, RFlex = np.concatenate(Flex), np.concatenate(RFlex)
    
    np.savetxt('Results/Dispatch_Flexible{}-{}-{}.csv'.format(scenario, stormZone, n_year), Flex, fmt='%f', delimiter=',', newline='\n', header='Flexible energy resources')
    np.savetxt('Results/Dispatch_RFlexible{}-{}-{}.csv'.format(scenario, stormZone, n_year), RFlex, fmt='%f', delimiter=',', newline='\n', header='Flexible energy resources')

    endtime = dt.datetime.now()
    print('.\nDispatch took', endtime - starttime)

    from Statistics import Information
    # Information(x, Flex, 3)

    return True

if __name__ == '__main__':

    # capacities = np.genfromtxt('CostOptimisationResults/Optimisation_resultx{}-None.csv'.format(scenario), delimiter=',')
    capacities = np.genfromtxt('Results/Optimisation_resultx{}-{}-{}.csv'.format(scenario, stormZone, n_year), delimiter=',')
    
    Analysis(capacities)