


#Global libs
import numpy as np


def RBF(u, v, g=0.01):
	rbf = np.sum(u**2, axis=1)[:,None] - np.sum(v**2, axis=1)[None,:]
	return np.exp(-g * np.abs(rbf))


def linearRBF(u, v, g=0.01):
	return np.dot(u, v.T) + RBF(u, v, g)