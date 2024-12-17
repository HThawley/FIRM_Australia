# To optimise the configurations of energy generation, storage and transmission assets
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

from scipy.optimize import differential_evolution
from datetime import datetime as dt

from Input import *
from Simulation import Reliability
from Network import Transmission
from Costs import pv_costs, onsw_costs, offsw_costs, ACgen_costs, phes_costs, hvdc_costs, hydro_purchase

def F(x):
    """This is the objective function."""

    S = Solution(x)

    Deficit = Reliability(S, flexible=np.zeros(intervals)) # Sj-EDE(t, j), MW
    Flexible = Deficit.sum() * resolution / years # MWh p.a.
    Hydro = Flexible + GBaseload.sum() * resolution / years # Hydropower & biomass: MWh p.a.
    PenHydro = max(0, Hydro - 20 * pow(10, 6)) # TWh p.a. to MWh p.a.

    Deficit = Reliability(S, flexible=np.ones(intervals) * CPeak.sum() * pow(10, 3)) # Sj-EDE(t, j), GW to MW
    PenDeficit = max(0, Deficit.sum() * resolution) # MWh

    TDC = Transmission(S) if scenario>=21 else np.zeros((intervals, len(DCloss))) # TDC: TDC(t, k), MW
    CDC = np.amax(abs(TDC), axis=0) * pow(10, -3) # CDC(k), MW to GW
    PenDC = max(0, CDC[6] - CDC6max) * pow(10, 3) # GW to MW

    cost = sum([
        sum(S.CPV)    * (pv_costs + ACgen_costs), 
        sum(S.COnsW)  * (onsw_costs + ACgen_costs), 
        sum(S.COffsW) * (offsw_costs + ACgen_costs),
        sum(S.CPHP)   * phes_costs[0], 
        S.CPHS * phes_costs[1], 
        #ignore PHES vom for now
        phes_costs[3],
        sum(CDC*hvdc_costs),
        Hydro * hydro_purchase
        ]) * pow(10,-9) # A$b
    
    loss = np.sum(abs(TDC), axis=0) * DCloss
    loss = loss.sum() * pow(10, -9) * resolution / years # PWh p.a.
    LCOE = cost / abs(energy - loss)

    Func = LCOE + PenHydro + PenDeficit + PenDC

    return Func

if __name__=='__main__':
    starttime = dt.now()
    print("Optimisation starts at", starttime)

    result = differential_evolution(
        func=F, 
        bounds=list(zip(lb, ub)), 
        tol=0,
        maxiter=args.i, 
        popsize=args.p, 
        mutation=(0.1, args.m),
        recombination=args.r,
        disp=True, 
        polish=False, 
        updating='deferred', 
        workers=-1,
        )

    np.savetxt('Results/Optimisation_resultx{}.csv'.format(scenario), result.x.reshape(1,-1), fmt='%s', delimiter=',')

    endtime = dt.now()
    print("Optimisation took", endtime - starttime)

    from Fill import Analysis
    Analysis(result.x)