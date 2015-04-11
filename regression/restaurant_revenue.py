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
 
DATA_PATH = "../data"
TRAIN_PATH = DATA_PATH + "/train.csv"
TEST_PATH = DATA_PATH + "/test.csv"

def load_all_data():
    
    labeled_data = load_data(TRAIN_PATH)
    unlabeled_data = load_data(TEST_PATH, labeled=False).X
    
    return preprocessData(labeled_data, unlabeled_data)
    
    
    
def main():
    
    labeled_data, unlabeled_data = load_all_data()
    
    print labeled_data.X
    

if __name__ == '__main__':
    main()