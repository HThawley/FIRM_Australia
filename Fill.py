# To simulate energy supply-demand balance based on long-term, high-resolution chronological data
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

from Input import * 
from Simulation import Reliability 

import datetime as dt
import numpy as np 

def Fill(solution):
    flexible = np.zeros(intervals, dtype=np.float64)
    Deficit, DeficitD = Reliability(solution, flexible=flexible)
    flex_cap = CPeak.sum()*1000
    
    fill = 0
    for t in range(intervals-1, -1, -1):
        d = Deficit[t]
        dd = DeficitD[t]
        if d > 0:
            flex = min(d, flex_cap - flexible[t]) 
            flexible[t] = flex
            if d-flex > 0:
                fill += (d-flex)/efficiency
        if dd > 0:
            flex = min(dd, flex_cap - flexible[t]) 
            flexible[t] += flex
            if dd-flex > 0:
                fill += (dd-flex)/efficiencyD
        if fill > 0:
            flex = min(fill, flex_cap - flexible[t]) 
            fill -= flex
            flexible[t] += flex
    
    Reliability(solution, flexible=flexible)
    return flexible
            
def Analysis(x):
    """Fill.Analysis(result.x)"""

    starttime = dt.datetime.now()
    print('Fill starts at', starttime)

    S = Solution(x)
    Flex = Fill(S)
    np.savetxt('Results/Dispatch_Flexible{}.csv'.format(scenario), Flex, fmt='%f', delimiter=',', newline='\n', header='Flexible energy resources')

    endtime = dt.datetime.now()
    print('Fill took', endtime - starttime)