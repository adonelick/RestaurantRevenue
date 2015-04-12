"""
Andrew Donelick
Alex Putman


Machine Learning - CS 158
Final Project: Predicting Restaraunt Revenue

This file is meant as the main file predicting and evaluating restaurant revenue

"""
from util import *
from preprocessing import *
from semi_supervised_knn import *
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

    
def main():
    
    labeled_data, unlabeled_data = load_all_data()
    clf = KNeighborsRegressor()
    clf.fit(labeled_data.X, labeled_data.y)

    generateOutputFile(clf, unlabeled_data)

    

if __name__ == '__main__':
    main()