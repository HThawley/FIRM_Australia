# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 10:20:24 2024

@author: u6942852
"""

import numpy as np 
from Input import *
import csv
import warnings
from datetime import datetime


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
        
        if it is not None:
            self.it = it
        
        if j is not None:  
            self.p_obj, self.p_x = self.eval_sample(base_x, inc, j)
            self.n_obj, self.n_x = self.eval_sample(base_x, -inc, j)
            
            self.p_grad = (self.p_obj - self.base_obj)/self.inc
            self.n_grad = (self.n_obj - self.base_obj)/self.inc
            
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
        

def gradient_descent(func, x0, bounds=None, maxiter=1000, disp=False, callback=None, incs=(10,1,0.1,0.01), convex=None):
    
    if bounds is not None:
        lb, ub = zip(*((pair for pair in bounds)))
        lb, ub = np.array(lb), np.array(ub)
        assert (x0 < lb).sum() == (x0 > ub).sum() == 0, "starting point outside of bounds"
    else:
        lb, ub = None, None
         
    base_x = x0.copy()
    ii, i = 0, 0
    inc = incs[ii]
    
    ws = working_solution(func, base_x, inc, ub, lb, convex)
    
    if disp is True:
        print(f"Optimisation starts: {datetime.now()}\n{'-'*40}")
    
    while i < maxiter:
        for j in range(len(base_x)):
            ws.update_self(base_x, inc, j, i)
            
            if disp is True: 
                print(f"iteration {i}: {ws.base_obj}")
            if callback is not None: 
                callback(ws)
            
            if ws.p_grad < 0: 
                base_x[j] += inc
                base_x = np.clip(base_x, lb, ub)
            elif ws.n_grad < 0:
                base_x[j] -= inc
                base_x = np.clip(base_x, lb, ub)
            else:
                try: 
                    ii+=1
                    inc = incs[ii]
                except IndexError():
                    break
        i+=1
            
    if j == len(incs): 
        termination = "Reached finest increment resolution"
    if i == maxiter:
        termination = "Reached maximum iterations"
    return ws, termination
    

def callback(ws):
    with open(cbfile, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([ws.it, ws.base_obj, 0] + list(ws.base_x))
        writer.writerow([ws.it, ws.p_obj, ws.inc] + list(ws.p_x))
        writer.writerow([ws.it, ws.n_obj, -ws.inc] + list(ws.n_x))

def init_callbackfile(n):
    with open(cbfile, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['iteration', 'objective', 'increment'] + ['steps']*n)
    
if __name__ == '__main__':
    x0 = np.array([ 10.18518519,   0.92592593,   0.92592593,   2.77777778,
            15.94650206,  15.74074074,   0.92592593,   2.77777778,
             2.77777778,   2.77777778,   0.72016461,   2.77777778,
             2.77777778,   2.77777778,   8.33333333,  20.74685416,
           270.91906722])
    
    lb = [0.]  * pzones + [0.]   * wzones + contingency   + [0.]
    ub = [50.] * pzones + [50.]  * wzones + [50.] * nodes + [5000.]
    
    cbfile = 'Results/GOptimHistory{}.csv'.format(scenario) 
    init_callbackfile(len(lb))

    from Optimisation import F
    
    ws, termination = gradient_descent(
        func=F,
        x0=x0,        
        bounds=list(zip(lb, ub)), 
        maxiter=20,
        disp=True,
        incs=(1,0.1,0.001,0.0001),
        callback=callback,
        convex=None,
        )

    print(termination)