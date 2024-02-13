# To optimise the configurations of energy generation, storage and transmission assets
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

import numpy as np 
import csv
import warnings
from datetime import datetime
from Setup import *



class working_solution:
    def __init__(self, func, base_x, inc, ub, lb, convex=None):
        self.objective = func
        self.base_x = base_x
        self.inc = inc
        self.lb, self.ub = lb, ub
        self.it = 0
        
        self = self.update_self(base_x, inc)

    def update_self(self, base_x, inc, j=None, it=None):
        self.base_x = base_x
        self.inc = inc
        self.base_obj = self.objective(base_x)
        
        self.j = j
        self.it = it
        
        if j is not None:  
            self.p_obj, self.p_x = self.eval_sample(base_x, inc, j)
            self.n_obj, self.n_x = self.eval_sample(base_x, -inc, j)
            
            self.p_step = (self.p_obj - self.base_obj)
            self.n_step = (self.n_obj - self.base_obj)
            
            self.p_grad = self.p_step/self.inc
            self.n_grad = self.n_step/self.inc

            if (self.p_grad < 0) and (self.n_grad < 0):
                assert convex is not True, "Problem is non-convex"
                self.p_grad = 0 if self.p_grad > self.n_grad else self.p_grad
                self.n_grad = 0 if self.p_grad < self.n_grad else self.n_grad
        
        return self

    def eval_sample(self, base_x, inc, i):
        samp_x = base_x.copy()
        samp_x[i] += inc
        if inc > 0: 
            samp_x = np.clip(samp_x, None, self.ub)
        else:
            samp_x = np.clip(samp_x, self.lb, None)
        
        return self.objective(samp_x), samp_x
        

def local_sampling(
        func, 
        x0, 
        bounds=None,
        maxiter=1000,
        disp=False, 
        callback=None, 
        incs=(10,1,0.1,0.01), 
        convex=True, 
        atol=1e-6, 
        rtol=-np.inf):
    
    def writerow_callback(ws):
        with open(callback, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([ws.it, ws.j, ws.base_obj, ws.inc, ws.p_step, ws.n_step] + list(ws.base_x))

    def init_callbackfile(n):
        with open(callback, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['iteration', 'dvar index', 'objective', 'increment', 'pos step obj', 'neg step obj'] + ['dvar']*n)
    
    if bounds is not None:
        lb, ub = zip(*((pair for pair in bounds)))
        lb, ub = np.array(lb), np.array(ub)
        assert (x0 < lb).sum() == (x0 > ub).sum() == 0, "starting point outside of bounds"
    else:
        lb, ub = None, None
    
    if callback is not None and args.x<2: 
        init_callbackfile(len(x0))
    
    base_x = x0.copy()
    ii, i = 0, 0
    inc = incs[ii]
    
    ws = working_solution(func, base_x, inc, ub, lb, convex)
    
    if disp is True:
        print(f"Optimisation starts: {datetime.now()}\n{'-'*40}")
    
    while i < maxiter:
        if disp is True: 
                print(f"iteration {i}: {ws.base_obj}")
        base_obj=ws.base_obj
        for j in range(len(base_x)):
            ws.update_self(base_x, inc, j, i)
            
            if callback is not None: 
                writerow_callback(ws)
            
            if ws.p_grad < 0: 
                base_x[j] += inc
                base_x = np.clip(base_x, lb, ub)
            elif ws.n_grad < 0:
                base_x[j] -= inc
                base_x = np.clip(base_x, lb, ub)

        dif = abs(ws.base_obj - base_obj) 
        if dif < atol or dif/base_obj < rtol:
            try: 
                ii+=1
                inc = incs[ii]
            except IndexError:
                termination = "Reached finest increment resolution"
                break
        i+=1

    if i == maxiter:
        termination = "Reached maximum iterations"
        
    return ws, termination
    
if __name__ == '__main__':
    x0 = np.genfromtext('costOptimisationResults/Optimisation_resultx{}-None.csv'.format(scenario))
    
    if args.his == 1: 
        cbfile = 'Results/GOptimHistory{}.csv'.format(scenario) 
    else:
        cbfile = None

    ws, termination = local_sampling(
        func=F,
        x0=x0,        
        bounds=bounds, 
        maxiter=50,
        disp=True,
        incs=[(10**n) for n in range(1, -6, -1)],
        callback=cbfile,
        convex=None,
        )

    print(termination)