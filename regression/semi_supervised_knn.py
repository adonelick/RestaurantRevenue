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
    def fit(self, L, U, max_it=1000, p1='euclidean',p2='chebyshev',pool_size=100):
        metrics = [p1,p2]
        # Initialize Training Sets
        Ls = [L, L]
        # Select pool of unlabeled data
        Upool_indexs = np.random.choice(len(U), pool_size, replace=False)
        Upool = [U[i] for i in Upool_indexs]
        
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
                print j
                Upool_ys = kNNs[j].predict(Upool)
                # get the neighbors of each unlabeled point - as indexs of the orig lists
                Upool_ns = kNNs[j].kneighbors(Upool, return_distance=False)
                
                deltas = []
                for r in range(len(Upool)):
                    alt_kNN = KNeighborsRegressor(n_neighbors=self.k,metric=metrics[j])
                    Lj_alt = Union(Ls[j], Upool[r], Upool_ys[r])
                    alt_kNN.fit(Lj_alt.X, Lj_alt.y)
                    
                    neighbors_indexs = Upool_ns[r]
                    neighbors = [Ls[j].X[n] for n in neighbors_indexs]
                    
                    kNN_n_ys = kNNs[j].predict(neighbors)
                    altkNN_n_ys = alt_kNN.predict(neighbors)
                    real_n_ys = [Ls[j].y[n] for n in neighbors_indexs]
                    delta = 0
                    for n in range(self.k):
                        orig_diff = real_n_ys[n] - kNN_n_ys[n]
                        alt_diff = real_n_ys[n] - altkNN_n_ys[n]
                        delta += orig_diff**2 - alt_diff**2
                    deltas.append(delta)
                    
                sorted_ds = sorted(deltas)[::-1]
                pi
                if sorted_ds[0] > 0:
                    highest = sorted_ds[0]
                    index = deltas.index(highest)
                    xj = Upool[index]
                    yj = Upool_ys[index]
                    
                    pi[j] = [(xj,yj)]
                    
                    np.delete(Upool, index)
            
            newLs = Ls
            for i in [0,1]:
                for px,py in pi[1-i]:
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
            Upool_indexs = np.random.choice(len(U), pool_size, replace=False)
            Upool = [U[i] for i in Upool_indexs]
        
        print kNNs[0].predict(U)
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
    for i in range(len(A.X)):
        if (A.X[i] == x).all() and (A.y[i] == y).all():
            contains = True
    
    newA = A
    m,n = A.X.shape
    if not contains:
        newX = [A.X[i] if i < m else x for i in range(m + 1)]
        newy = [A.y[i] if i < m else y for i in range(m + 1)]
        newA.X = np.array(newX)
        newA.y = np.array(newy)
    return newA
    
        
            
            
    