# Step-by-step analysis to decide the dispatch of flexible energy resources
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

from Input import *
from Simulation import Reliability
from CoSimulation import Resilience

import datetime as dt
from multiprocessing import Pool, cpu_count

def Flexible(idx, x):
    """Energy source of high flexibility"""
    S = Solution(x)
    startidx, endidx = idx
    if testing:
        endidx = startidx+intervals
    Fcapacity = CPeak.sum() * 1000 # GW to MW
    flexible = Fcapacity * np.ones(endidx - startidx)
    
    Deficit, DeficitD = Reliability(S, flexible=flexible, start=startidx, end=endidx, output=False) # Sj-EDE(t, j), MW
    Deficiti_1 = (Deficit + DeficitD).sum() * resolution
    
    for i in range(0, 350, timestep):
        flexible[i: i+timestep] = 0
        Deficit, DeficitD = Reliability(S, flexible=flexible, start=startidx, end=endidx, output=False) # Sj-EDE(t, j), MW
        Deficiti = (Deficit + DeficitD).sum() * resolution
        if Deficiti > Deficiti_1:
            flexible[i: i+timestep] = Fcapacity
        
    Reliability(S, flexible, start=startidx, end=endidx, output=True)
    flexible = np.maximum(flexible - S.Spillage, 0)
    return flexible


def Analysis(x):
    """Dispatch.Analysis(result.x)"""
    starttime = dt.datetime.now()
    print('Dispatch starts at', starttime)

    idxs = [(
        int((24 / resolution) * (dt.datetime(year, 1, 1) - dt.datetime(firstyear, 1, 1)).days),
        int((24 / resolution) * (dt.datetime(year+1, 1, 1) - dt.datetime(firstyear, 1, 1)).days))
        for year in range(firstyear, finalyear + 1)]

    # Multiprocessing
    with Pool(processes=min(cpu_count(), (finalyear - firstyear + 1))) as pool:
        instances  = [(idx, x) for idx in idxs]
        Dispresult = pool.starmap(Flexible, instances)
        
    Dispresult = [flex for flex in Dispresult]
    Flex = np.concatenate(Dispresult)
    
    np.savetxt('Results/Dispatch_Flexible'+suffix, Flex, fmt='%f', delimiter=',', newline='\n', header='Flexible energy resources')

    endtime = dt.datetime.now()
    print('Dispatch took', endtime - starttime)

    from Statistics import Information, DeficitInformation

    Information(x, Flex, resilience=False)
    
    return True

if __name__ == '__main__':
    
    # capacities = np.genfromtxt('CostOptimisationResults/Optimisation_resultx{}-None.csv'.format(scenario), delimiter=',')
    capacities = np.genfromtxt('Results/Optimisation_resultx'+suffix, delimiter=',')

    Analysis(capacities)
