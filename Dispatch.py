# Step-by-step analysis to decide the dispatch of flexible energy resources
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

from Input import *
from Simulation import Reliability
from CoSimulation import Resilience

import datetime as dt
from multiprocessing import Pool, cpu_count

def Flexible(year, x):
    """Energy source of high flexibility"""
    print(year, end=', ')
    S = Solution(x)

    startidx = int((24 / resolution) * (dt.datetime(year, 1, 1) - dt.datetime(firstyear, 1, 1)).days)
    endidx = int((24 / resolution) * (dt.datetime(year+1, 1, 1) - dt.datetime(firstyear, 1, 1)).days)

    Fcapacity = CPeak.sum() * pow(10, 3) # GW to MW
    flexible = Fcapacity * np.ones(endidx - startidx)

    for i in range(0, endidx - startidx, timestep):
        flexible[i: i+timestep] = 0
        Deficit, DeficitD = Reliability(S, flexible=flexible, start=startidx, end=endidx, output=True) # Sj-EDE(t, j), MW
        if (Deficit + DeficitD)[i:].sum() * resolution > 0.1:
            flexible[i: i+timestep] = Fcapacity
            
    flexible = np.clip(flexible - S.Spillage, 0, None)
    return pd.DataFrame([[0, year, flexible]])

    
def RFlexible(year, x):
    """Energy source of high flexibility"""
    print('r-', year, end=', ')
    S = Solution(x)
    
    startidx = int((24 / resolution) * (dt.datetime(year, 1, 1) - dt.datetime(firstyear, 1, 1)).days)
    endidx = int((24 / resolution) * (dt.datetime(year+1, 1, 1) - dt.datetime(firstyear, 1, 1)).days)

    Fcapacity = CPeak.sum() * pow(10, 3) # GW to MW
    flexible = Fcapacity * np.ones(endidx - startidx)
    
    for i in range(0, endidx - startidx, timestep):
        flexible[i: i+timestep] = 0
        Deficit, DeficitD, RDeficit, RDeficitD = Resilience(S, flexible=flexible, start=startidx, end=endidx, output='deficits') # Sj-EDE(t, j), MW
        if (RDeficit + RDeficitD)[i:].sum() * resolution > 0.1:
            flexible[i: i+timestep] = Fcapacity
            
    flexible = np.clip(flexible - S.RSpillage, 0, None)
    return pd.DataFrame([[1, year, flexible]])
    
    
def CRFlexible(year, x):
    """Energy source of high flexibility"""
    print('cr-', year, end=', ')
    S = Solution(x)
    
    startidx = int((24 / resolution) * (dt.datetime(year, 1, 1) - dt.datetime(firstyear, 1, 1)).days)
    endidx = int((24 / resolution) * (dt.datetime(year+1, 1, 1) - dt.datetime(firstyear, 1, 1)).days)

    Fcapacity = CPeak.sum() * pow(10, 3) # GW to MW
    flexible = Fcapacity * np.ones(endidx - startidx)
    
    for i in range(0, endidx - startidx, timestep):
        flexible[i: i+timestep] = 0
        Deficit, DeficitD, RDeficit, RDeficitD = Resilience(S, flexible=flexible, start=startidx, end=endidx, output='deficits') # Sj-EDE(t, j), MW
        if (RDeficit + RDeficitD)[i:].sum() * resolution > 0.1:
            flexible[i: i+timestep] = Fcapacity
            
    flexible = np.clip(flexible - S.RSpillage, 0, None)
    return pd.DataFrame([[2, year, flexible]])


def Analysis(x, flex=True):
    """Dispatch.Analysis(result.x)"""
    costCapacities = np.genfromtxt('CostOptimisationResults/Optimisation_resultx{}-None.csv'.format(scenario), delimiter=',')
    if flex:
        starttime = dt.datetime.now()
        print('Dispatch starts at', starttime)
        print('Dispatch works on: ', end='')

        # Multiprocessing
        with Pool(processes=min(cpu_count(), 3*(finalyear - firstyear + 1))) as pool:
            instances  = [(year, x) for year in range(firstyear, finalyear + 1)]
            instancesCR = [(year, costCapacities) for year in range(firstyear, finalyear + 1)]
    
            Dispresult = pool.starmap(Flexible, instances)
            DispresultR = pool.starmap(RFlexible, instances)
            DispresultCR = pool.starmap(CRFlexible, instancesCR)
            
        result = list(Dispresult) + list(DispresultR) + list(DispresultCR)
        
        result = pd.concat(result)
        result = result.sort_values(1)
        
        Flex, RFlex, CRFlex = (np.array(np.concatenate(list(result.loc[result[0]==i, 2]))) for i in range(3))
        
        np.savetxt('Results/Dispatch_Flexible{}-{}-{}.csv'.format(scenario, stormZone, n_year), Flex, fmt='%f', delimiter=',', newline='\n', header='Flexible energy resources')
        np.savetxt('Results/Dispatch_RFlexible{}-{}-{}.csv'.format(scenario, stormZone, n_year), RFlex, fmt='%f', delimiter=',', newline='\n', header='Flexible energy resources')
        np.savetxt('Results/Dispatch_CRFlexible{}-{}-{}.csv'.format(scenario, stormZone, n_year), CRFlex, fmt='%f', delimiter=',', newline='\n', header='Flexible energy resources')

        endtime = dt.datetime.now()
        print('.\nDispatch took', endtime - starttime)

    else: 
        Flex = np.genfromtxt('Results/Dispatch_Flexible{}-{}-{}.csv'.format(scenario, stormZone, n_year), delimiter=',')
        RFlex = np.genfromtxt('Results/Dispatch_RFlexible{}-{}-{}.csv'.format(scenario, stormZone, n_year), delimiter=',')
        CRFlex = np.genfromtxt('Results/Dispatch_CRFlexible{}-{}-{}.csv'.format(scenario, stormZone, n_year), delimiter=',')

    from Statistics import Information, DeficitInformation, verifyDispatch
    try: verifyDispatch(costCapacities, np.genfromtxt('CostOptimisationResults//Dispatch_Flexible{}-None.csv'.format(scenario), delimiter=','))
    except AssertionError: print(f'{"="*50}\nS{scenario}, Z{stormZone}, C deficit assertion failed')
    try: verifyDispatch(x, Flex)
    except AssertionError: print(f'{"="*50}\nS{scenario}, Z{stormZone}, deficit assertion failed')
    try: verifyDispatch(x, RFlex, resilience=True)    
    except AssertionError: print(f'{"="*50}\nS{scenario}, Z{stormZone}, R deficit assertion failed')


    Information(x, Flex, resilience=False)
    # Information(x, RFlex, resilience=True)
    DeficitInformation(costCapacities, CRFlex, 1)
    
    return True

if __name__ == '__main__':
    
    # capacities = np.genfromtxt('CostOptimisationResults/Optimisation_resultx{}-None.csv'.format(scenario), delimiter=',')
    capacities = np.genfromtxt('Results/Optimisation_resultx{}-{}-{}.csv'.format(scenario, stormZone, n_year), delimiter=',')

    Analysis(capacities, True)