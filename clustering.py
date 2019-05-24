'''

'''

import numpy as np

def kmeans(X, means, epsilon=0, max_it=100):
    ''' kmeans(X, means, epsilon=0) -> yields Y,  means, delta, step
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
    N = X.shape[0]
    K = means.shape[0]
    W = np.zeros([N,K])
    Y = np.zeros(N)
    delta = None
    
    for step in range(max_it):
        for k, m in enumerate(means):
            W[:,k] = np.sum((X-m)**2, axis=1)
        
        Y = np.argmin(W, axis=1)
        
        tmp = means.copy()
        for k in range(K):
            means[k,:] = np.mean(X[Y==k,:], axis=0)
        
        delta = np.sum((tmp - means)**2)
        yield Y, means, delta, step #yield for mean update
        
        if delta <= epsilon:
            break
    pass


def init_kmeans(X, K, mode='free'):
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
            W = np.sum((X-means[k-1])**2, axis=1) #get the distances
            p = np.cumsum(W/np.sum(W)) #spread probabillities from 0 to 1
            i = p.searchsorted(np.random.rand(), 'right') #pick next center to random distance
            means[k,:] = X[i,:]
    else: #free
        pass
    return means
    
