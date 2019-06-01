'''
Author: Gerald Baulig
'''

#Global libs
import numpy as np

__all__ = ['RBF', 'linearRBF', 'none']

none = None

def linear(u, v, params={}):
    return np.dot(u, v.T)


def RBF(u, v, params={}):
    g = float(params['gamma']) if 'gamma' in params else 1.0
    rbf = np.sum(u**2, axis=1)[:,None] - np.sum(v**2, axis=1)[None,:]
    return np.exp(-g * rbf**2)


def linearRBF(u, v, params={}):
    return linear(u, v.T) + RBF(u, v, params)