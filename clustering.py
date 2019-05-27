'''

'''

import numpy as np


def square_mag(a, b):
    return np.sum((a-b)**2, axis=1)

def kmeans(X, means, epsilon=0.0, max_it=100):
    ''' kmeans(X, means, epsilon=0.0) -> yields Y,  means, delta, step
    A generator for simple KMeans clustering.
    
    Usage:
        See demo_kmeans.py
    Args:
        X: The data N-by-d, where
            N=num datapoints
            d=dimensions
        means: The cluster centers.
        epsilon: Convergence threshold.
            (default=0)
    Yields:
        Y: The labels or cluster association.
        means: The updated cluster centers.
        delta: The distance to the update step.
        step: The iteration step.
    '''
    N = X.shape[0]      #N: Size of the dataset
    K = means.shape[0]  #K: Number of clusters
    W = np.zeros([N,K]) #W: Distance (Weight) matrix
    Y = np.zeros(N)     #Y: labels
    delta = None
    
    for step in range(max_it):
        for k, m in enumerate(means):
            W[:,k] = square_mag(X, m) #calc square distances
        
        Y = np.argmin(W, axis=1) #find closest
        
        tmp = means.copy()
        for k in range(K):
            means[k,:] = np.mean(X[Y==k,:], axis=0) #calc the means
        means[np.isnan(means)] = 0.0 #prevent the apocalypse
        
        delta = np.sum((tmp - means)**2) #calc the delta of change
        yield Y, means, delta, step #yield for mean update
        
        if delta <= epsilon:
            break #converge if change event is lower-equal than epsilon
    pass


def init_kmeans(X, K, mode='zeros'):
    ''' init_kmeans(X, K, mode='free') -> means
    Initialize K cluster means on dataset X.
    
    Args:
        X: The dataset.
        K: The number of cluster means.
        mode: The mode how to initialize the cluster means.
            zeros = All means start at zero.
                (Requires centered dataset)
            uniform = Uniform random distributed.
                (Requires centered dataset)
            normal = Normal random distributed.
                (Requires centered dataset)
            select = Selects random points of the dataset.
            kmeans++ = Selects a random point and rearranges the
                probabillities of selecting the next point according
                to the distance.
    Returns:
        means: The cluster means.
    '''
    N = X.shape[0]
    d = X.shape[1]
    means = np.zeros([K,d])
    if mode=='select':
        means = X[np.random.choice(range(N), K),:]
    if mode=='uniform':
        pass
    elif mode=='normal':
        pass
    elif mode=='kmeans++':
        #https://stackoverflow.com/questions/5466323/how-exactly-does-k-means-work
        means[0,:] = X[np.random.choice(range(N), 1),:] #pick one
        for k in range(1,K):
            W = square_mag(X, means[k-1]) #get the distances
            p = np.cumsum(W/np.sum(W)) #spread probabillities from 0 to 1
            i = p.searchsorted(np.random.rand(), 'right') #pick next center to random distance
            means[k,:] = X[i,:]
    else: #zeros
        pass
    return means


def dbscan(X, n, r):
    N = X.shape[0]
    Y = np.zeros(N)
    r = r**2
    
    def pick_next():
        ''' pick_next() -> yields x
        A generator to pick next unlabeled point.
        
        Yields:
            i: The next unlabeled datapoint.
        '''
        while True:
            i = Y.searchsorted(0, 'right') #'searchsorted' is faster than 'any' or 'where'
            if i != -1:
                yield X[i]
            else:
                break
        pass
    
    def find_neighbors(x):
        i = square_mag(X, x) <= r
        return i

    def cluster():
        pass
    
    for C, x in enumerate(pick_next()):
        i = find_neighbors(x)
        if sum(i) >= n:
            Y[i] = C+1
            
