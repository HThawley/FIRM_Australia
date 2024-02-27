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
import datetime as dt
from tqdm import tqdm

from Setup import *

ndim = sidx+1

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

def signs_of_array(arr, tol=1e-10):
    arr[np.abs(arr)<tol] = 0 
    return np.sign(arr)

class hypercube():
    def __init__(self, centre, f, bounds, parent_f, parent_n):
        self.centre = centre
        self.parent_f = parent_f
        self.f = float(f)
        self.bounds = bounds
        self.rdif = self.f/self.parent_f
        self.adif = self.f-self.parent_f
        self.n = parent_n + 1
        # volume = 10^vol_indx - this method avoid floating point error for tiny volumes in high dim
        self.vol_indx = (np.log10(self.bounds.ub-self.bounds.lb)).sum()
        

    def is_same(self, cube):
        ub_same = (self.bounds.ub == cube.bounds.ub).product() == 1 
        lb_same = (self.bounds.lb == cube.bounds.lb).product() == 1 
        c_same = (self.centre == cube.centre).product() == 1
        f_same = self.f == cube.f
        return ub_same and lb_same and c_same and f_same

    def borders(self, cube, tol=1e-10):
        """ 
        """
        # directions where the domains of each cube touch
        touch = (((self.bounds.ub - cube.bounds.lb) >= -tol) * 
                 ((cube.bounds.ub - self.bounds.lb) >= -tol))

        # the domains in each direction touch   
        if not (touch.sum() == ndim):
            return False

        # in ndim-1 directions the directions' domains overlap (either perfectly, or one inside another)
        overlap = (signs_of_array(self.bounds.ub - cube.bounds.ub, tol) ==
                   signs_of_array(cube.bounds.lb - self.bounds.lb, tol))
        if not (overlap.sum() == ndim-1):
            return False

        # in exactly one direction domains do not overlap (although they may touch)
        if not ((~overlap).sum() == 1):
            return False
        
        # adjacent (ub=lb or lb=ub) (higher OR lower) in exactly one dimension
        adjacency = ((np.abs(self.bounds.ub - cube.bounds.lb) < tol) +
                     (np.abs(self.bounds.lb - cube.bounds.ub) < tol)) 
        if not (adjacency.sum() == 1):
            return False

        # Direction of adjacency is the direction not overlapping
        if not ((adjacency == ~overlap).prod() == 1):
            return False
        return True

def eval_hypercube(centre, f, bounds, parent_f, parent_n):
    return hypercube(centre, f, bounds, parent_f, parent_n)

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
    population: int = 1,
    min_vol_indx:float=-np.inf,
):
  
    def _func_wrap(x, args=None):
        x = np.asarray(x)
        if args is None:
            f = func(x)
        else:
            f = func(x, *args)
        return f
            
    def _algorithm():

        def divide_cube(cube):
            indcs = np.array(list(product([True,False], repeat=len(cube.centre))))
            
            centres = generate_centres(cube, indcs)
            bounds = generate_bounds(cube, indcs)
            
            if vectorizable is True: 
                f_values = _func_wrap((norm*centres).T, args)
            else: 
                f_values = np.array([_func_wrap(norm*xn, args) for xn in centres])
            
            with Pool(processes=min(cpu_count(), bounds.shape[0])) as processPool:
                cubes = processPool.starmap(
                    eval_hypercube, 
                    [(centres[k], f_values[k], bounds[k], cube.f, cube.n) for k in range(bounds.shape[0])],
                    )

            return cubes

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
                indx
                ) for indx in indcs])
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
        parent = hypercube(norm_bounds.ub/2, _func_wrap(norm*norm_bounds.ub/2, args)[0], norm_bounds, np.inf, -1)
        parents = [parent]
        archive = np.array([], dtype=int)

        while i < maxiter:
            it_start = dt.datetime.now()
            # split all cubes to be split from previous iteration
            new_cubes = np.array([divide_cube(parent) for parent in parents]).flatten()

            # all cubes which do not have any children  
            childless = np.concatenate((new_cubes, archive))

            # generate pairs of list-index and cost
            fs = np.array([(j, cube.f) for j, cube in enumerate(childless)])
            # sort list indices by cost
            fs = fs[fs[:,1].argsort(),0]
            # get list-indicies of the best {population} cubes by cost 
            best = np.array(fs[:min(population, len(fs))], dtype=int)

            # find list-indices of cubes adjacent to best in the new generation of cubes
            new_accepted = np.array([j for b in childless[best] for j, cube in enumerate(childless) if b.borders(cube) and cube.vol_indx>min_vol_indx], dtype=int)
            
            # combine new and archived cubes to be split next iteration
            parents = childless[np.unique(np.concatenate((best, new_accepted)))]

            # get list-indices of childless cubes which are not to be split
            to_arch = np.array(list(set(range(len(childless))) - set(new_accepted)), dtype=int)
    
            # update archive
            archive = childless[to_arch]

            print(f'i = {i}: #cubes = {len(parents)}. Took: {dt.datetime.now()-it_start}')
            i+=1
        
        best = [(cube.f, norm*cube.centre) for cube in parents]
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

    
if __name__ == '__main__':

    starttime = dt.datetime.now()
    print("Optimisation starts at", starttime)

    result = direct(
        func=F_v, 
        args=(True,),
        population=args.p,
        # eps=0.5,
        bounds=list(zip(lb, ub)), 
        # maxfun=args.i*len(lb),
        maxiter=args.i, 
        min_vol_indx=-3*ndim, #1 MW in each direction
        # callback=cb,
        # vol_tol=0,
        vectorizable=True,
        )

    endtime = dt.datetime.now()
    print("Optimisation took", endtime - starttime)

    print(result)