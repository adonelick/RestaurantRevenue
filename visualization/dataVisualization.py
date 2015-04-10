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
import time
import csv

DATA_PATH = "../data"
TRAIN_PATH = DATA_PATH + "/train.csv"
TEST_PATH = DATA_PATH + "/test.csv"

def main():

    
    with open(TEST_PATH, 'rb') as dataFile:

        trainReader = csv.reader(dataFile, delimiter=',')

        # Search the file for any values which are non-numeric, 
        # and create the necessary lookup tables from 
        # the non-numeric values to numbers

        cityNames = {}

        firstRow = True
        for row in trainReader:
            if firstRow:
                firstRow = False
                continue

            if not cityNames.has_key(row[2]):
                cityNames[row[2]] = len(cityNames)

            month, day, year = ''.join(row[1][0:2]), ''.join(row[1][3:5]), ''.join(row[1][6:10])
            #print time.mktime(time.strptime(month + " " + day + " " + year, "%m %d %Y"))

            if row[3] == "Big Cities":
                print 1
            else:
                print 0





if __name__ == '__main__':
    main()
