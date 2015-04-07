"""
Andrew Donelick
Alex Putman

Machine Learning - CS 158
Final Project: Predicting Restaraunt Revenue

This file is meant to help us get a feel for the data:
we would like to debug the file I/O, visualize the features,
and give us a sense of what regression techniques might 
be useful later in the project.

"""

import numpy as np
from matplotlib import pyplot as plt
import csv

DATA_PATH = "../data"
TRAIN_PATH = DATA_PATH + "/train.csv"
TEST_PATH = DATA_PATH + "/test.csv"

def main():

    cityNames = set()
    with open(TEST_PATH, 'rb') as dataFile:


        trainReader = csv.reader(dataFile, delimiter=',')
        for row in trainReader:
            cityNames.add(row[2])

    print len(cityNames)


if __name__ == '__main__':
    main()
