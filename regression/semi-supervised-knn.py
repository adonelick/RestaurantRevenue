"""
Andrew Donelick
Alex Putman

Machine Learning - CS 158
Final Project: Predicting Restaraunt Revenue

This file is meant as one possible semi supervised regressor for 
K nearest neighbors.

"""

import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from operator import add

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
    
    def fit(self, L, U, max_it=1000, p1='euclidean',p2='mahalanobis',pool_size=100):
        metrics = [p1,p2]
        # Initialize Training Sets
        Ls = [L, L]
        # Select pool of unlabeled data
        Upool = np.random.choice(U, pool_size, replace=False)
        
        # Create the two kNN regressors
        kNNs = [KNeighborsRegressor(n_neighbors=self.k,metric=m) for m in metrics]
        # train regressors on both sets
        for i in [0,1]:
            kNNs[i].fit(Ls[i].X, Ls[i].y)
        
        # repeat for max_it rounds
        for i in range(max_it):
            
            # keep list of changes to Ls
            pi = [[],[]]
            # for each training and regressor set
            for j in [0,1]:
                Upool_ys = kNNs[j].predict(Upool)
                # get the neighbors of each unlabeled point - as indexs of the orig lists
                Upool_ns = kNNs[j].kneighbors(Upool, return_distance=False)
                
                deltas = []
                for r in range(len(Upool)):
                    alt_kNN = KNeighborsRegressor(n_neighbors=self.k,metric=metrics[j])
                    Lj_alt = Union(L[j], Upool[r], Upool_ys[r])
                    alt_kNN.fit(Lj_alt.X, Lj_alt.y)
                    
                    neighbors_indexs = Upool_ns[r]
                    neighbors = [L[j].X[n] for n in neighbors_indexs]
                    
                    kNN_n_ys = kNNs[j].predict(neighbors)
                    altkNN_n_ys = alt_kNN.predict(neighbors)
                    real_n_ys = [L[j].y[n] for n in neighbors_indexs]
                    delta = 0
                    for n in range(self.k):
                        orig_diff = real_n_ys[n] - kNN_n_ys[n]
                        alt_diff = real_n_ys[n] - altkNN_n_ys[n]
                        delta += orig_diff**2 - alt_diff**2
                    deltas.append(delta)
                    
                sorted_ds = reversed(sorted(deltas))
                pi
                if sorted_ds[0] > 0:
                    highest = sorted_ds[0]
                    index = deltas.index(highest)
                    xj = Upool[index]
                    yj = Upool_ys[index]
                    
                    pi[j] = [(xj,yj)]
                    
                    Upool.remove(xj)
            
            newLs = Ls
            for i in [0,1]:
                for px,py in pi[1-i]:
                    newLs[i] = Union(newLs[i],px,py)
            # if no changes need to be made, we have converged 
            if newLs == Ls:
                break
            
            # else make changes, retrain, and replinesh untrained pool
            Ls = newLs
            for i in [0,1]:
                kNNs[i].fit(Ls[i].X, Ls[i].y)
            Upool = np.random.choice(U, pool_size, replace=False)
            
        self.h1 = kNNs[0]
        self.h2 = kNNs[0]
        
    def predict(self, X):
        
        h1_p = self.h1.predict(X)
        h2_p = self.h2.predict(X)
        
        results = [(x + y)/2.0 for x, y in zip(h1_p, h2_p)]
        
        return results
        
def Union(A,x,y):
    # get shape of A and B
    contains = False
    for i in range(len(A.X)):
        if A.X[i] == x and A.y[i] == y:
            contains = True
    
    newA = A
    if not contains:
        newA.X += x
        newA.y += y
    return newA
    
        
            
            
    