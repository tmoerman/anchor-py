import numpy as np
import math
from numpy.random import RandomState


def Random_Projection(M, new_dim: int, prng: RandomState):
    """
    Project the columns of the matrix M into the lower dimension new_dim.

    :param M:
    :param new_dim:
    :param prng:
    :return:
    """

    old_dim = M[:, 0].size
    p = np.array([1./6, 2./3, 1./6])
    c = np.cumsum(p)
    random_doubles = prng.random_sample(new_dim*old_dim)

    R = np.searchsorted(c, random_doubles)
    R = math.sqrt(3) * (R - 1)
    R = np.reshape(R, (new_dim, old_dim))
    
    return np.dot(R, M)