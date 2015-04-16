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

class EnsembleCotrainer:

    def __init__(self, n_estimators=200, n_iterations=100):
        
        self.n_estimators = n_estimators
        self.n_iterations = n_iterations
        self.estimators = []


    def fit(self, labeledData, unlabeledData):
        
        self.estimators = []
        for i in xrange(self.n_estimators):
            print i

            clf = KNeighborsRegressor()
            clf.fit(labeledData.X, labeledData.y)
            n, d = unlabeledData.shape

            indices = np.random.choice(np.array(range(n)), size=100, replace=False)
            unlabeledTrainingData = unlabeledData[indices]

            for j in xrange(self.n_iterations):

                labels = clf.predict(unlabeledTrainingData)

                newTrainingData_X = np.append(labeledData.X, unlabeledTrainingData, axis=0)
                newTrainingData_y = np.append(labeledData.y, labels)

                clf = KNeighborsRegressor()
                clf.fit(newTrainingData_X, newTrainingData_y)

            self.estimators.append(clf)

    def predict(self, X):

        n, d = X.shape
        labels = np.zeros((n))
        for clf in self.estimators:
            labels += clf.predict(X)

        labels /= (1.0 * self.n_estimators)
        return labels


