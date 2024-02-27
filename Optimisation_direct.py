# To optimise the configurations of energy generation, storage and transmission assets
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

from scipy.optimize import direct

import datetime as dt
import csv

from Setup import *

def callback(xk):
    with open(cbfile, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([F(xk)] + list(xk))

def init_callbackfile(n):
    with open(cbfile, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['objective'] + ['dvar']*n)


#%%

if __name__=='__main__':
    starttime = dt.datetime.now()
    print("Optimisation starts at", starttime)

    lb = [0.]  * pzones + [0.]   * wzones + contingency   + [0.]
    ub = [50.] * pzones + [50.]  * wzones + [50.] * nodes + [5000.]

    if bool(args.his) is True:
        cbfile = 'Results/OptimisationHistory{}.csv'.format(scenario)
        init_callbackfile(len(ub))
        cb = callback
    else: 
        cb = None
        
    result = direct(
        func=F_d, 
        eps=1e-3,
        bounds=list(zip(lb, ub)), 
        maxfun=args.i*len(lb),
        maxiter=args.i, 
        callback=cb,
        vol_tol=0,
        locally_biased=False,
        )

    with open('Results/Optimisation_resultx{}.csv'.format(scenario), 'a', newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(result.x)

    endtime = dt.datetime.now()
    print("Optimisation took", endtime - starttime)

    from Dispatch import Analysis
    Analysis(result.x)