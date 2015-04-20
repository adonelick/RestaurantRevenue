"""
Andrew Donelick
Alex Putman

Machine Learning - CS 158
Final Project: Predicting Restaraunt Revenue

This file is meant as one possible semi supervised regressor for 
K nearest neighbors.

"""

import numpy as np
from util import *
import heapq
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
"""
This is a form of semi supervised learning using the k Nearest Neighbors 
approach with co-training. The formula that was implmented was found in
a paper by Zhi-Hua Zhou and Ming Li seen here:
    
    http://ijcai.org/papers/0689.pdf
    
"""
class Knn_semi:
    
    def __init__(self, k=5):
        self.k = k
        self.h1 = None
        self.h2 = None
    
    """
    fit function
        
    fits the data to the two models h1 and h2
    
    Inputs:
        L           -   labeled data set of type data set with X and y values
        U           -   unlabeled data set in list form
        max_it      -   the max number of iterations before giving up on convergence
        p1, p2      -   the two distance metrics used for training kNNs 
                        (should be different)
        pool_size   -   the numbered of unlabeled data points that will try and be
                        fitted and added to the training set. 
    """
    def fit(self, L, U, max_it=1000, p1='euclidean',p2='mahalanobis',pool_size=100):
        metrics = [p1,p2]
        # Initialize Training Sets
        L1 = Data(np.copy(L.X), np.copy(L.y))
        L2 = Data(np.copy(L.X), np.copy(L.y))
        Ls = [L1, L2]
        # Select pool of unlabeled data
        Upool_indexs = np.random.choice(len(U), pool_size, replace=False)
        Upool = [U[i] for i in Upool_indexs]
        
        
        # Create the two kNN regressors
        kNNs = []
        for m in metrics:
            r = None
            if m == 'mahalanobis':
                pca = PCA()
                pca.fit(L.X)
                v = pca.get_covariance()
                r = KNeighborsRegressor(n_neighbors=self.k,metric=m, V=v)
            else:
                r = KNeighborsRegressor(n_neighbors=self.k,metric=m)
            kNNs.append(r)
        # train regressors on both sets
        for i in [0,1]:
            kNNs[i].fit(Ls[i].X, Ls[i].y)
        
        # repeat for max_it rounds
        for i in range(max_it):
            print i
            # keep list of changes to Ls
            pi = [[],[]]
            # for each training and regressor set
            for j in [0,1]:
                #print j
                Upool_ys = kNNs[j].predict(Upool)
                # get the neighbors of each unlabeled point - as indexs of the orig lists
                Upool_ns = kNNs[j].kneighbors(Upool, return_distance=False)
                
                deltas = []
                for r in xrange(len(Upool)):
                    Lj_alt = Union(Ls[j], Upool[r], Upool_ys[r])
                    alt_kNN = None
                    m = metrics[j]
                    if m == 'mahalanobis':
                        pca.fit(Lj_alt.X)
                        v = pca.get_covariance()
                        alt_kNN = KNeighborsRegressor(n_neighbors=self.k,metric=m, V=v)
                    else:
                        alt_kNN = KNeighborsRegressor(n_neighbors=self.k,metric=m)
                    alt_kNN.fit(Lj_alt.X, Lj_alt.y)
                    
                    neighbors_indexs = Upool_ns[r]
                    neighbors = [Ls[j].X[n] for n in neighbors_indexs]
                    
                    kNN_n_ys = kNNs[j].predict(neighbors)
                    altkNN_n_ys = alt_kNN.predict(neighbors)
                    real_n_ys = [Ls[j].y[n] for n in neighbors_indexs]
                    delta = 0
                    for n in xrange(self.k):
                        orig_diff = real_n_ys[n] - kNN_n_ys[n]
                        alt_diff = real_n_ys[n] - altkNN_n_ys[n]
                        delta += orig_diff**2 - alt_diff**2
                    deltas.append(delta)
                    
                sorted_ds = sorted(deltas)[::-1]
                if sorted_ds[0] > 0:
                    highest = sorted_ds[0]
                    index = deltas.index(highest)
                    xj = Upool[index]
                    yj = Upool_ys[index]
                    
                    pi[j] = [(xj,yj)]
                    
                    uIndex = U.tolist().index(xj.tolist())
                    np.delete(U, uIndex)
            
            newLs = Ls
            replenishCount = 0
            for i in [0,1]:
                for px,py in pi[1-i]:
                    replenishCount += 1
                    newLs[i] = Union(newLs[i],px,py)
            # if no changes need to be made, we have converged 
            empty = True
            for a in pi:
                if a:
                    empty = False
            
            if empty:
                break
            
            # else make changes, retrain, and replinesh untrained pool
            Ls = newLs
            for i in [0,1]:
                kNNs[i].fit(Ls[i].X, Ls[i].y)
            #Upool_indexs = np.random.choice(len(U), replenishCount, replace=False)
            #Upool_addition = [U[i] for i in Upool_indexs]
            #Upool = np.append(Upool, Upool_addition, axis=0)
            Upool_indexs = np.random.choice(len(U), pool_size, replace=False)
            Upool = [U[i] for i in Upool_indexs]
        
        #print kNNs[0].predict(U)
        self.h1 = kNNs[0]
        self.h2 = kNNs[1]
        
    """
    Predict function
    
    Input:
        X   -   set of labels that you want to predict from the 
                kNN semi supervised model
    
    Output:
        list of predictions
    """
    def predict(self, X):
        
        h1_p = self.h1.predict(X)
        h2_p = self.h2.predict(X)
        
        results = [(x + y)/2.0 for x, y in zip(h1_p, h2_p)]
        
        return results
       
class SemiSupervisedLearner:
    
    def __init__(self, learner, k=5):
        self.k = k
        self.learner = learner
        self.model = None
    
    """
    fit function
        
    fits the data to the two models h1 and h2
    
    Inputs:
        L           -   labeled data set of type data set with X and y values
        U           -   unlabeled data set in list form
        maxIt       -   the maximum number of iterations before giving up
                        on convergence
        pool_size   -   the numbered of unlabeled data points that will try and be
                        fitted and added to the training set. 
        **kwargs    -   named arguments for the learner that was given on class
                        initialization
    """
    def fit(self, L, U, maxIt=1000, poolSize=100, wSize=10, **kwargs):
        
        # Initialize Training Sets
        L = Data(np.copy(L.X), np.copy(L.y))
        
        # Select pool of unlabeled data
        UpoolIndexs = np.random.choice(len(U), poolSize, replace=False)
        Upool = [U[i] for i in UpoolIndexs]
        
        # Create the regressor
        model = self.learner(**kwargs)
        
        # train regressors on labeled data
        model.fit(L.X, L.y)
        
        # repeat for max_it rounds
        for i in range(maxIt):
            print i
            # keep list of changes to Ls
            pi = []
                
            UpoolYs = model.predict(Upool)
            # get the neighbors of each unlabeled point - as indexs of the orig lists
            kNN = KNeighborsRegressor(n_neighbors=self.k)
            kNN.fit(L.X, L.y)
            
            UpoolNDistances = [sum(ns) for ns in kNN.kneighbors(Upool)[0]]
            W = heapq.nsmallest(wSize, [(k, t) for t, k in enumerate(UpoolNDistances)])
            W = [w[1] for w in W]
            Wpool = [Upool[r] for r in W]
            WNeighbors = kNN.kneighbors(Wpool, return_distance=False)
            RMSEs = []
            newX = []
            newY = []
            for r in range(wSize):

                neighborsIndexs = WNeighbors[r]
                neighbors = [L.X[n] for n in neighborsIndexs]
                
                neighborsYs = model.predict(neighbors)
                avgY = sum(neighborsYs)/float(self.k)
                x = Upool[W[r]]
                newX.append(x)
                newY.append(avgY)
            
            
            for x, y in zip(newX, newY):
                # L combined with the neighbors of each u in the Upool
                altL = Union(L, x, y)
                
                # create a model based on this altL
                altModel = self.learner(**kwargs)
                altModel.fit(altL.X, altL.y)
                
                altY = altModel.predict(newX)
                
                rmse = mean_squared_error(newY, altY)
                
                RMSEs.append(rmse)
                
            sortedErrors = sorted(RMSEs)
            lowest = sortedErrors[0]
            index = W[RMSEs.index(lowest)]
            bestX = Upool[index]
            bestY = UpoolYs[index]
            
            L = Union(L, bestX, bestY)
            
            uIndex = U.tolist().index(bestX.tolist())
            m, n = U.shape
            U = np.delete(U, (uIndex), axis=0)
            
            
            model.fit(L.X, L.y)
            UpoolIndexs = np.random.choice(len(U), poolSize, replace=False)
            Upool = [U[i] for i in UpoolIndexs]
        
        #print kNNs[0].predict(U)
        print L.X
        print L.y
        self.model = model
        
    """
    Predict function
    
    Input:
        X   -   set of labels that you want to predict from the 
                kNN semi supervised model
    
    Output:
        list of predictions
    """
    def predict(self, X):
        return self.model.predict(X)
       
""" 
Union function

A U {(x,y)}

Input:
    A   -   A data set with X and y values
    x   -   new set of labels you want to be unioned into A
    y   -   the true value for the x wanting to be unioned into A
    
Output:
    data set with X and y values
""" 
def Union(A,x,y):
    # get shape of A and B
    contains = False
    for i in xrange(len(A.X)):
        if (A.X[i] == x).all() and (A.y[i] == y).all():
            contains = True
    
    newA = Data(np.copy(A.X), np.copy(A.y))
    #newA = A
    if not contains:
        newA.X = np.append(A.X, [x], axis=0)
        newA.y = np.append(A.y, y)

    return newA
    
        
            
            
    