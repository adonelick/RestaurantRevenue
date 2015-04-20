"""
Andrew Donelick
Alex Putman


Machine Learning - CS 158
Final Project: Predicting Restaraunt Revenue

This file is meant as the main file predicting and evaluating restaurant revenue

"""
from util import *
from evaluation import *
from preprocessing import *
from semi_supervised_knn import *
from EnsembleCotrainer import *
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import scale, normalize

def test():

    labeled_data, unlabeled_data = load_all_data()
    bestData = load_data(BEST_SUBMISSION_PATH)
    bestY = bestData.y[1:].astype(np.float)

    #bestY = scale(bestY)
    #labeled_data.y = scale(labeled_data.y)

    for x in xrange(5, 6):

        clf = DecisionTreeRegressor()
        clf.fit(labeled_data.X, labeled_data.y)

        newData = Data()
        newData.X = np.append(labeled_data.X, unlabeled_data[0:95000], axis=0)
        newData.y = np.append(labeled_data.y, bestY[0:95000])

        for i in xrange(x):
            clf = RandomForestRegressor()
            clf.fit(newData.X, newData.y)

            n, d = unlabeled_data.shape

            indices = np.random.choice(np.array(range(n)), size=5000, replace=False)
            labels = clf.predict(unlabeled_data[indices])
            newData.X = np.append(labeled_data.X, unlabeled_data[indices], axis=0)
            newData.y = np.append(labeled_data.y, labels)
        
            print rmse(bestY, clf.predict(unlabeled_data))

        saveRevenues(clf.predict(unlabeled_data))

    
def main():
    
    labeled_data, unlabeled_data = load_all_data()
    bestData = load_data(BEST_SUBMISSION_PATH)
    bestY = bestData.y[1:].astype(np.float)

    # for numClassifiers in xrange(1, 2):
    #     for iterations in xrange(10, 200, 10):
    #         predictions = np.zeros((100000))
    #         for i in xrange(numClassifiers):
    #             clf = Knn_semi()
    #             clf.fit(labeled_data, unlabeled_data, max_it=iterations)
    #             predictions += clf.predict(unlabeled_data)

    #         predictions /= (1.0 * numClassifiers)

    #         print numClassifiers, iterations, rmse(bestY, predictions)


    # test()

    clf = EnsembleCotrainer()
    clf.fit(labeled_data, unlabeled_data)
    labels = clf.predict(unlabeled_data)

    print rmse(bestY, labels)

    #clf = DecisionTreeRegressor()
    #clf.fit(labeled_data.X, labeled_data.y)

    saveRevenues(labels)
    #generateOutputFile(clf, unlabeled_data)

    

if __name__ == '__main__':
    main()