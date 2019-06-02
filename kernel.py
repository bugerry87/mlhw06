'''
A set of kernel functions and helpers.

Author: Gerald Baulig
'''

#Global libs
import numpy as np

__all__ = ['linear', 'RBF', 'linearRBF', 'none']

none = None


def parse_params(arg_string):
    ''' parse_params(arg_string) -> dict
    Parses an argument string to a dictionary.
    E.g.: 'gamma=1.0, sigma=2.0'
    
    Args:
        arg_string: A comma seperated argument string.
    Returns:
        params: The related dictionary of key=value.
    '''
    if arg_string:
        arg_string = arg_string.strip()
        params = args.params.split(',')
        params = [p.split('=') for p in params]
        params = {p[0]:p[1] for p in params}
        return params
    else:
        return {}


def linear(u, v, params={}):
    ''' linear(u, v, params={}) -> kernel
    A simple linear kernel.
        k(u,v) = u * v.T
    
    Args:
        u: The left hand values.
        v: The right hand values.
        params: No params required.
    Returns:
        kernel: A linear kernel
    '''
    return np.dot(u, v.T)


def RBF(u, v, params={}):
    ''' RBF(u, v, params={}) -> kernel
    A simple Radial Basis Function kernel.
        k(u,v) = exp(-g*(u - v.T)^2)
    
    Args:
        u: The left hand values.
        v: The right hand values.
        params:
            gamma: The variance factor.
    Returns:
        kernel: A RBF kernel
    '''
    g = float(params['gamma']) if 'gamma' in params else 1.0
    rbf = np.sum(u**2, axis=1)[:,None] - np.sum(v**2, axis=1)[None,:]
    return np.exp(-g * np.abs(rbf))


def linearRBF(u, v, params={}):
    ''' RBF(u, v, params={}) -> kernel
    A combination of linear and Radial Basis Function kernel.
        k(u,v) = u * v.T + exp(-g*(u - v.T)^2)
    
    Args:
        u: The left hand values.
        v: The right hand values.
        params:
            gamma: The variance factor.
    Returns:
        kernel: A linearRBF kernel
    '''
    return linear(u, v.T) + RBF(u, v, params)