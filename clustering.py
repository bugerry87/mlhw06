'''

'''

import numpy as np
from itertools import chain


def square_mag(a, b):
    ''' square_mag(a, b) -> squared magnitude
    Calculates the row-wise squared magnitude.
    This function is preferably used for distance comparisons,
    because taking a square root has a high computation time.
        np.sum((a-b)**2, axis=1)
    
    Args:
        a: numpy.ndarray
        b: numpy.ndarray
    Returns:
        The row-wise squared magnitude of a and b.
    '''
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


KMEANS_INIT_MODES = ['zeros', 'select', 'uniform', 'normal', 'kmeans++']
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
        #guided by https://stackoverflow.com/questions/5466323/how-exactly-does-k-means-work
        means[0,:] = X[np.random.choice(range(N), 1),:] #pick one
        for k in range(1,K):
            W = square_mag(X, means[k-1]) #get the distances
            p = np.cumsum(W/np.sum(W)) #spread probabillities from 0 to 1
            i = p.searchsorted(np.random.rand(), 'right') #pick next center to random distance
            means[k,:] = X[i,:]
    else: #zeros
        pass
    return means


def dbscan(X, points, radius):
    ''' dbscan(X, points, radius) -> yields Y, x, step
    Simple DBScan clustering with one path following agent.
    Takes long computation time, because almost every point needs to be checked.
    Otherwise it could not complete the pathfinding.
    The utilization of the radius is very low.
    
    Args:
        X:
    '''
    #guided by https://github.com/chrisjmccormick/dbscan
    N = X.shape[0]
    Y = np.zeros(N)
    radius = radius**2
    C = 1
    step=0
    
    def pick_next():
        ''' pick_next() -> yields i
        A generator to pick next unlabeled point.
        
        Yields:
            i: The next index of unlabeled datapoint.
        '''
        while True:
            i = np.nonzero(Y==0)[0]
            if i.size == 0:
                break
            else:
                yield i[0]
        pass
    
    def scan(i):
        ''' scan(i) -> yields n
        The core function of dbscan.
        Check the given point under index i to be:
        core-point, border-point, or noice.
        
        '''
        n = square_mag(X, X[i]) <= radius
        if sum(n) >= points:
            n[i] = False #Do not count the current core point.
            n = np.logical_and(n, np.logical_or(Y==0, Y==-1)) #check for unprocessed points
            Y[n] = C
            return np.where(n)[0]
        else:
            Y[n] = -1
            return np.ndarray(0)
        pass
    
    for i in pick_next():
        F = [i]
        for i in F:
            F.extend(scan(i).tolist())
            step += 1
            yield Y, X[i], step
        C += 1
    pass


def spectral(X, K, epsilon=0, sigma=1, mode='default'):
    '''
    '''
    N = X.shape[0]
    d = X.shape[1]
    K = K-1
    
    def weights(X):
        W = 0
        for i in range(d):
            x = X[:,i]
            W += (x - x[:,None])**2 # [:,None] ~ Transpose!
        W = np.exp(-W/sigma)
        return W
        
    def degrees(W):
        D = np.sum(W, axis=1)*np.eye(N)
        return D
    
    def laplacian(W, D, mode='default'):
        L = D-W
        if mode == 'shi':
            D = np.linalg.inv(D)
            return D*L*D
        elif mode == 'jodran':
            D = np.linalg.inv(D)
            return D*L
        else:
            return L
    
    def cluster(eigvec):
        if K == 0:
            return np.zeros([N,1])
        
        C = np.zeros([N, K])
        for k in range(0,K):
            C[:,k] = eigvec[k+1] > 0
        return C
    
    W = weights(X)
    D = degrees(W)
    L = laplacian(W, D, mode)
    eigval, eigvec = np.linalg.eig(L)
    C = cluster(eigvec)
    
    return L, eigval, eigvec, C