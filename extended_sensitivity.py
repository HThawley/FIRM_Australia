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
        
        self = self.update_self(base_x, inc)

    def update_self(self, base_x, inc):
        self.base_x = base_x
        self.inc = inc
        self.base_obj = self.objective(base_x)
        
        self.p_objs = np.array([self.eval_sample(base_x, inc, i) for i in range(len(base_x))])
        self.n_objs = np.array([self.eval_sample(base_x, -inc, i) for i in range(len(base_x))])
        
        self.p_grads = (self.p_objs - self.base_obj)/self.inc
        self.n_grads = (self.n_objs - self.base_obj)/self.inc
        
        if self.p_grads.min() < 0 or self.n_grads.min() < 0: 
            self.p_mask, self.n_mask = self.p_grads<0, self.n_grads<0
            
            if (b_mask:=(self.p_mask * self.n_mask)).sum() != 0:
                assert convex is not True, "Problem is non-convex"
                self.p_mask[(self.p_grads>self.n_grads)*b_mask] = False
                self.n_mask[(self.p_grads<self.n_grads)*b_mask] = False
        
        return self

    def eval_sample(self, base_x, inc, i):
        samp_x = base_x.copy()
        samp_x[i] += inc
        if inc > 0: 
            samp_x = np.clip(samp_x, None, self.ub)
        else:
            samp_x = np.clip(samp_x, self.lb, None)
        
        return self.objective(samp_x) 
        

def gradient_descent(func, x0, bounds, maxiter, disp, callback, incs, convex):
    
    lb, ub = zip(*((pair for pair in bounds)))
    lb, ub = np.array(lb), np.array(ub)
    assert (x0 < lb).sum() == (x0 > ub).sum() == 0, "starting point outside of bounds"
         
    base_x = x0.copy()
    j, i = 0, 0
    inc = incs[j]
    
    ws = working_solution(func, base_x, inc, ub, lb, convex)
    
    if disp is True:
        print(f"Optimisation starts: {datetime.now()}\n{'-'*40}")
    
    while i < maxiter:

            ws.update_self(base_x, inc)
            
            if disp is True: 
                print(f"iteration {i}: {ws.base_obj}")
            if callback is not None: 
                callback(ws)
            
            if ws.p_grads.min() < 0 or ws.n_grads.min() < 0: 
                base_x[ws.p_mask] += (inc/10.)
                base_x[ws.n_mask] -= (inc/10.)
                base_x = np.clip(base_x, lb, ub)
            
            else:
                try: 
                    j+=1
                    inc = incs[j]
                except IndexError():
                    break
            i+=1
    True
    if j == len(incs): 
        termination = "Reached finest increment resolution"
    if i == maxiter:
        termination = "Reached maximum iterations"
    return ws, termination
    

def callback(ws):
    with open(cbfile, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([ws.base_obj, 0] + list(ws.base_x))
        writer.writerow([-1, ws.inc] + list(ws.p_objs))
        writer.writerow([-1, -ws.inc] + list(ws.n_objs))

def init_callbackfile(n):
    with open(cbfile, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['objective', 'increment'] + ['steps']*n)
    
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