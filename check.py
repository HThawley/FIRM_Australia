import numpy as np
from Input import * 

x0 = np.genfromtxt('CostOptimisationResults/Optimisation_resultx{}.csv'.format(scenario), delimiter = ',', dtype=float)
x1 = x0 + 0.1*np.ones(x0.shape)
x2 = x0 + -0.1*np.ones(x0.shape)

x = np.stack((x0, x1, x2)).T

S = Solution(x, vectorized=True)
flexible = np.ones((intervals,1)) * CPeak.sum()

from Simulation import Reliability
d = Reliability(S, flexible)
print(d)