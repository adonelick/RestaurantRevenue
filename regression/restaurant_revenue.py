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
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import scale

def test():

    labeled_data, unlabeled_data = load_all_data()
    bestData = load_data(BEST_SUBMISSION_PATH)
    bestY = bestData.y[1:].astype(np.float)

    #bestY = scale(bestY)
    #labeled_data.y = scale(labeled_data.y)

    for x in xrange(10, 11):

        clf = AdaBoostRegressor()
        clf.fit(labeled_data.X, labeled_data.y)

        newData = Data()
        newData.X = np.append(labeled_data.X, unlabeled_data[0:85000], axis=0)
        newData.y = np.append(labeled_data.y, bestY[0:85000])

        for i in xrange(x):
            clf = AdaBoostRegressor()
            clf.fit(newData.X, newData.y)

            labels = clf.predict(unlabeled_data)
            newData.X = np.append(labeled_data.X, unlabeled_data, axis=0)
            newData.y = np.append(labeled_data.y, labels)
        
            print rmse(bestY, clf.predict(unlabeled_data))

        saveRevenues(clf.predict(unlabeled_data))

    
def main():
    
    labeled_data, unlabeled_data = load_all_data()
    # bestData = load_data(BEST_SUBMISSION_PATH)
    # bestY = bestData.y[1:].astype(np.float)

    # for numClassifiers in xrange(1, 2):
    #     for iterations in xrange(10, 200, 10):
    #         predictions = np.zeros((100000))
    #         for i in xrange(numClassifiers):
    #             clf = Knn_semi()
    #             clf.fit(labeled_data, unlabeled_data, max_it=iterations)
    #             predictions += clf.predict(unlabeled_data)

    #         predictions /= (1.0 * numClassifiers)

    #         print numClassifiers, iterations, rmse(bestY, predictions)


    #test()

    clf = SemiSupervisedLearner(DecisionTreeRegressor)
    clf.fit(labeled_data, unlabeled_data, maxIt=10, poolSize=1000)
    p = clf.predict(unlabeled_data)
    print p
    #compare_to_best(p)
    #  C=1e7, gamma=0.0001
    #saveRevenues(predictions)
    generateOutputFile(clf, unlabeled_data)

    
def compare_to_best(predictions):
    bestData = load_data(BEST_SUBMISSION_PATH)
    bestY = bestData.y[1:].astype(np.float)
    print mean_squared_error(bestY, predictions)
    
if __name__ == '__main__':
    main()