'''

'''

#Global libs
import numpy as np


#Local libs
import kernel


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


KMEANS_INIT_MODES = ('zeros', 'select', 'uniform', 'normal', 'kmeans++')
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
    means = np.zeros((K,d))
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


def kmeans(X, means, epsilon=0.0, max_it=1000):
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
    W = np.zeros((N,K)) #W: Distance (Weight) matrix
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


def kernel_trick(gram, C):
    N = gram.shape[0]
    K = C.shape[1]
    A = np.sum(C,axis=0)
    ones = np.ones((N, K))
    
    W = np.matmul(gram * np.eye(N), ones)
    W -= 2*(np.matmul(gram, C) / A)
    W += np.matmul(ones, np.matmul(np.matmul(C.T, gram), C) * np.eye(K))
    W /= (A**2)
    return W


def kernel_kmeans(X, means, epsilon=0.0, max_it=1000):
    N = X.shape[0]      #N: Size of the dataset
    K = means.shape[0]  #K: Number of clusters
    W = np.zeros((N,K)) #W: Distance (Weight) matrix
    C = np.zeros((N,K)) #C: Associations
    Y = np.zeros(N)     #Y: Labels
    delta = None
    
    #init
    for k, m in enumerate(means):
        W[:,k] = square_mag(Gram, m) #calc square distances
    Y = np.argmin(W, axis=1) #find closest
    C = np.nonzero(Y)
    
    for step in range(max_it):
        
        
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


LAPLACIAN_MODES = ('default', 'shi', 'jordan')
class Spectral:
    '''
    '''

    def __init__(self, X=None, gamma=1, epsilon=0, mode='default'):
        self.__gamma = float(gamma) if float(gamma) > 0 else 1
        self.__epsilon = float(epsilon)
        self.__mode = mode if mode else 'default'
        self.__N = 0
        self.__d = 0
        self.__W = 0
        self.__D = 0
        self.__L = 0
        self.__eigval = 0
        self.__eigvec = 0
        self.set(X, gamma, epsilon, mode)
        
    def set(self, X=None, gamma=None, epsilon=None, mode=None):
        recalc = False
        if isinstance(X, np.ndarray):
            self.__X = X
            self.__N = X.shape[0]
            self.__d = X.shape[1]
            recalc = True
        
        if gamma != None:
            self.__gamma = gamma
            recalc = True
        
        if recalc:
            self.__W = kernel.RBF(self.__X, self.__X, gamma)
        
        if epsilon != None:
            self.__epsilon = epsilon
            recalc = True
        
        if recalc:
            if self.__epsilon:
                self.__D = np.sum(self.__W > self.__epsilon, axis=1)*np.eye(self.__N)
            else:
                self.__D = np.sum(self.__W, axis=1)*np.eye(self.__N)
        
        if mode != None:
            self.__mode = mode
            recalc = True
        
        if recalc:
            L = self.__D - self.__W
            if mode == 'shi':
                D = np.sqrt(D)
                D = np.linalg.inv(self.__D)
                self.__L = D*L*D
            elif mode == 'jodran':
                D = np.linalg.inv(self.__D)
                self.__L = D*L
            elif mode == 'default':
                self.__L = L
            else:
                raise ValueError("Unknown mode!")
        
            self.__eigval, self.__eigvec = np.linalg.eig(L)
        pass
    
    def cluster(self, K, mode='select'):
        X = self.__eigvec[:,0:K]
        means = init_kmeans(X, K, mode)
        return kmeans(X, means)

    @property
    def X(self): return self.__X
    
    @property
    def N(self): return self.__N
    
    @property
    def d(self): return self.__d
    
    @property
    def gamma(self): return self.__gamma
    
    @property
    def epsilon(self): return self.__epsilon
    
    @property
    def mode(self): return self.__mode
    
    @property
    def W(self): return self.__W
    
    @property
    def D(self): return self.__D
    
    @property
    def L(self): return self.__L
    
    @property
    def eigval(self): return self.__eigval
    
    @property
    def eigvec(self): return self.__eigvec
