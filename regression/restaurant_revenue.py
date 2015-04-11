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

# Paths for the data files
DATA_PATH = "../data"
TRAIN_PATH = DATA_PATH + "/train.csv"
TEST_PATH = DATA_PATH + "/test.csv"
PREPROCESSED_TRAIN_PATH = DATA_PATH + "/preprocessed_train.csv"
PREPROCESSED_TEST_PATH = DATA_PATH + "/preprocessed_test.csv"


def load_all_data():
    """
    Loads in the labeled (training) and unlabeled (testing)
    data. The function automatically preprocesses the 
    data according to the preprocessing file.

    :return: (tuple of Data, 2-d numpy array) Preprocessed training/test data
    """

    try:
        trainData = load_data(PREPROCESSED_TRAIN_PATH)
        testData = load_data(PREPROCESSED_TEST_PATH, labeled=False).X

        trainData.X = trainData.X.astype(np.float)
        trainData.y = trainData.y.astype(np.float)
        testData = testData.astype(np.float)

    except Exception, e:   
        labeled_data = load_data(TRAIN_PATH)
        unlabeled_data = load_data(TEST_PATH, labeled=False).X
        trainData, testData = preprocessData(labeled_data, unlabeled_data, 
                                             PREPROCESSED_TRAIN_PATH, 
                                             PREPROCESSED_TEST_PATH)
    
    return trainData, testData
    
    
    
def main():
    
    labeled_data, unlabeled_data = load_all_data()
    
    print unlabeled_data
    

if __name__ == '__main__':
    main()