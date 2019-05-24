'''

'''

import numpy as np

def kmeans(X, means, epsilon=0):
    ''' kmeans(X, means, epsilon=0) -> yields Y,  means, delta
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
    '''
    N = X.shape[0]
    K = means.shape[0]
    W = np.zeros([N,K])
    Y = np.zeros(N)
    delta = None
    
    while True:
        for k, m in enumerate(means):
            W[:,k] = np.sum((X-m)**2, axis=1)
        
        Y = np.argmin(W, axis=1)
        
        tmp = means.copy()
        for k in range(K):
            means[k,:] = np.mean(X[Y==k,:], axis=0)
        
        delta = np.sum((tmp - means)**2)
        yield Y, means, delta #yield for mean update
        
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
        means[0,:] = X[np.random.choice(range(N), 1),:]
        for k in range(K):
            W = np.sum((X-m)**2, axis=1)
            p = W/np.sum(W) #probabillity
            cumprobs = probs.cumsum()
            r = scipy.rand()
            for j,p in enumerate(cumprobs):
                if r < p:
                    i = j
                    break
                C.append(X[i])
            return C
        
    else: #free
        pass
    return means
    
