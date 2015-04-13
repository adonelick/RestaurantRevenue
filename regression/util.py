# author      : Jessica Wu, Andrew Donelick, Alex Putman
# date        : 04/11/2015
# description : ML utilities

# python libraries
import os
import csv

# numpy libraries
import numpy as np

# matplotlib libraries
import matplotlib as mpl
import matplotlib.pyplot as plt

import preprocessing
import util
######################################################################
# global settings
######################################################################

mpl.lines.width = 2
mpl.axes.labelsize = 14

# Paths for the data files
DATA_PATH = "../data"
TRAIN_PATH = DATA_PATH + "/train.csv"
TEST_PATH = DATA_PATH + "/test.csv"
PREPROCESSED_TRAIN_PATH = DATA_PATH + "/preprocessed_train.csv"
PREPROCESSED_TEST_PATH = DATA_PATH + "/preprocessed_test.csv"
PREDICTIONS_PATH = DATA_PATH + "/submission.csv"


######################################################################
# classes
######################################################################

class Data :
    
    def __init__(self, X=None, y=None) :
        """Data class"""
        
        # n = number of examples, d = dimensionality
        self.X = X    # nxd array
        self.y = y    # rank-1 array (think row vector)
    
    def load(self, filename, labeled=True) :
        """Load csv file into X array of features and y array of labels"""
        
        # determine filename
        dir = os.path.dirname(util.__file__)
        f = os.path.join(dir, filename)
        
        # load data
        with open(f, 'r') as fid :
            data = np.genfromtxt(fid, delimiter=",", dtype=None)
        
        # separate features and labels
        if labeled:
            self.X = data[:,:-1]
            self.y = data[:,-1] # rank-1 array (think row vector)
        else:
            self.X = data[:,]
    
    def plot(self) :
        """Plot features and labels"""
        pos = np.nonzero(self.y > 0)  # matlab: find(y > 0)
        neg = np.nonzero(self.y < 0)  # matlab: find(y < 0)
        plt.plot(self.X[pos,0], self.X[pos,1], 'b+', markersize=5)
        plt.plot(self.X[neg,0], self.X[neg,1], 'ro', markersize=5)
        plt.show()

# helper functions
def load_data(filename, labeled=True) :
    """Load csv file into Data class"""
    data = Data()
    data.load(filename, labeled)
    return data

def load_all_data():
    """
    Loads in the labeled (training) and unlabeled (testing)
    data. The function automatically preprocesses the 
    data according to the preprocessing file.

    :return: (tuple of Data, 2-d numpy array) Preprocessed training/test data
    """

    try:
        # First, attempt to load the preprocessed training
        # and testing data from disk
        trainData = load_data(PREPROCESSED_TRAIN_PATH)
        testData = load_data(PREPROCESSED_TEST_PATH, labeled=False).X

        trainData.X = trainData.X.astype(np.float)
        trainData.y = trainData.y.astype(np.float)
        testData = testData.astype(np.float)

    except Exception, e:
        # If it doesn't exist yet, create it from the original data
        labeled_data = load_data(TRAIN_PATH)
        unlabeled_data = load_data(TEST_PATH, labeled=False).X
        trainData, testData = preprocessing.preprocessData(labeled_data, unlabeled_data, 
                                                           PREPROCESSED_TRAIN_PATH, 
                                                           PREPROCESSED_TEST_PATH)
    
    return trainData, testData

def generateOutputFile(regressor, unlabeledData):
    """
    Generates a file for submission to the Kaggle 
    website. It has the format:

    Id, Prediction
    0, value0
    1, value1
    ...
    N, valueN

    :param regressor: a class which has been trained to predict 
                      revenue given a new, unlabeled sample
    :param unlabeledData: (2-d numpy array) unlabeled samples
    :return: (2-d numpy array) revenue predictons
    """

    predictions = np.array([["Id", "Prediction"]])
    revenues = regressor.predict(unlabeledData)

    for i, revenue in enumerate(revenues):

        row = np.array([str(i), str(revenue)])
        predictions = np.append(predictions, [row], axis=0)
    
    
    dir = os.path.dirname(util.__file__)
    fs = os.path.join(dir, PREDICTIONS_PATH)
    with open(fs, 'wb') as f:
        csv.writer(f, delimiter=',').writerows(predictions)

    return predictions


