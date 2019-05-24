

import numpy as np

def kmeans(X, means, epsilon):
	'''
	kmeans(X, means, epsilon=None) -> yield means, labels
	'''
	changes = True
	while changes:
		for m in means:
			delta = (X - m)**2
	yield
	