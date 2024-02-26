# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 14:13:22 2024

@author: u6942852
"""

from typing import (
    Any, Callable, Iterable, Optional, Tuple, Union
)

import numpy as np
import numpy.typing as npt

from scipy.optimize import Bounds, OptimizeResult
from scipy.optimize._constraints import old_bound_to_new

from multiprocessing import Pool, cpu_count
from itertools import product

# from Setup import *


NoneType = type(None)

ERROR_MESSAGES = (
    "Number of function evaluations done is larger than maxfun={}",
    "Number of iterations is larger than maxiter={}",
    "u[i] < l[i] for some i",
    "maxfun is too large",
    "Initialization failed",
    "There was an error in the creation of the sample points",
    "An error occured while the function was sampled",
    "Maximum number of levels has been reached.",
    "Forced stop",
    "Invalid arguments",
    "Out of memory",
)

SUCCESS_MESSAGES = (
    ("The best function value found is within a relative error={} "
     "of the (known) global optimum f_min"),
    ("The volume of the hyperrectangle containing the lowest function value "
     "found is below vol_tol={}"),
    ("The side length measure of the hyperrectangle containing the lowest "
     "function value found is below len_tol={}"),
)




def direct(
    func: Callable[[npt.ArrayLike, Tuple[Any]], float],
    bounds: Union[Iterable, Bounds],
    *,
    args: tuple = (),
    eps: float = 1e-4,
    maxfun: Union[int, None] = None,
    maxiter: int = 1000,
    locally_biased: bool = True,
    f_min: float = -np.inf,
    f_min_rtol: float = 1e-4,
    vol_tol: float = 1e-16,
    len_tol: float = 1e-6,
    callback: Optional[Callable[[npt.ArrayLike], NoneType]] = None,
    vectorizable: bool = True,
):
  
    def _func_wrap(x, args=None):
        x = np.asarray(x)
        if args is None:
            f = func(x)
        else:
            f = func(x, *args)
        return f
    
    class hypercube():
        def __init__(self, centre, f, bounds, parent_f):
            self.centre = centre
            self.parent_f = parent_f
            self.f = f
            self.bounds = bounds
            self.rdif = self.f/self.parent
            self.adif = self.f-self.parent
            self.accept = self.adif < 0 
            
    def _algorithm():
            
        def eval_hypercube(coords, parent, func):
            return hypercube(coords, parent, func)
        
        def divide_cube(cube):
            
            indcs = np.array(list(product([True,False], repeat=len(cube.centre))))
            
            centres = generate_centres(cube, indcs)
            bounds = generate_bounds(cube, indcs)
            if vectorizable is True: 
                f_values = _func_wrap(centres.T)
            else: 
                f_values = np.array([_func_wrap(xn, *args) for xn in centres])
            
            with Pool(process=min(cpu_count(), bounds.shape[0])) as processPool:
                cubes = processPool.starmap(
                    eval_hypercube, 
                    [(centres[n], f_values[n], bounds[n], cube.f) for n in range(bounds.shape[0])],
                    )
                
            return [cube for cube in cubes if cube.accept is True]

        def generate_bounds(cube, indcs):
            def _bound(ub, lb, indx):
                inv = ~indx
                base = np.ones((indx.shape[0], 2))
                base[indx] = ub[indx]
                base[inv] = lb[inv]
                
                return Bounds(base[:,0], base[:,1])
            
            bounds = np.array([_bound(
                np.concatenate(
                    (cube.centre.reshape(-1,1),
                     cube.bounds.ub.reshape(-1,1)),
                    axis=1),
                np.concatenate(
                    (cube.bounds.lb.reshape(-1,1),
                     cube.centre.reshape(-1,1)),
                    axis=1),
                    
                indx) for indx in indcs])
            return bounds

        def generate_centres(cube, indcs):
            def _coord(arr1, arr2, indx):
                inv = ~indx
                base = np.ones(indx.shape)
                base[indx] = arr1[indx]
                base[inv] = arr2[inv]
                return base
            
            centres = np.array([_coord(
                (cube.bounds.ub + cube.centre)/2,
                (cube.centre + cube.bounds.lb)/2,
                indx) for indx in indcs])
            return centres
        
        i=0
        parent = hypercube(norm_bounds.ub/2, _func_wrap(norm*norm_bounds/2, *args), norm_bounds, np.inf)
        parents = [parent]
        
        while i < maxiter:
            accepted_cubes = [divide_cube(parent) for parent in parents]
            if len(accepted_cubes) == 0:
                break
            parents = [cubes for cube_list in accepted_cubes for cubes in cube_list]
                        
            i+=1
        
        best = [(cube.f, cube.x) for cube in parents]
        best_f = np.array([item[0] for item in best])
        best_x = np.concatenate([item[1].reshape(1, -1) for item in best], axis = 0)
        
        best_f, best_x = best_f.max(), best_x[best_f.argmax()]
        
        return best_x, best_f
        
    
    # convert bounds to new Bounds class if necessary
    if not isinstance(bounds, Bounds):
        if isinstance(bounds, list) or isinstance(bounds, tuple):
            lb, ub = old_bound_to_new(bounds)
            bounds = Bounds(lb, ub)
        else:
            message = ("bounds must be a sequence or "
                       "instance of Bounds class")
            raise ValueError(message)
    
    lb = np.ascontiguousarray(bounds.lb, dtype=np.float64)
    ub = np.ascontiguousarray(bounds.ub, dtype=np.float64)
    

    
    norm = bounds.ub - bounds.lb
    norm_bounds = Bounds([0]*len(norm), [1]*len(norm))
    
    x_best, f_best =_algorithm()
    
    return np.asarray(x_best), f_best

    
            
               

        
    
        