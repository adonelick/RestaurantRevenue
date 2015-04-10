"""
Andrew Donelick
Alex Putman

Machine Learning - CS 158
Final Project: Predicting Restaraunt Revenue

This file is meant to clean up the data we are given.

"""

import numpy as np
from util import *
import time

def preprocessData(trainData, testData, saveLocation):
    """
    Preprocesses the train and test data files to 
    remove strings, time stamps, and other non-numerical
    values. These preprocessed data files are then saved
    in the location specified, and returned by the function.
    """

    # Search the file for any values which are non-numeric, 
    # and create the necessary lookup tables from 
    # the non-numeric values to numbers

    cityNames = {}

    firstRow = True
    for row in testData:
        if firstRow:
            firstRow = False
            continue

        if not cityNames.has_key(row[2]):
            cityNames[row[2]] = len(cityNames)


    newTrainData = Data(np.copy(trainData.X), np.copy(trainData.y))
    for i, row in enumerate(trainData.X):
        if i is 0:
            continue
        # Converts the city into a number
        realCity = 0

        # Converts the time from a string into a number
        month, day, year = ''.join(row[1][0:2]), ''.join(row[1][3:5]), ''.join(row[1][6:10])
        realTime = time.mktime(time.strptime(month + " " + day + " " + year, "%m %d %Y"))

        cityType = 0
        if row[3] == 'Big Cities':
            cityType = 1
            
        row4 = 0
        if row[4] == 'IL':
            row4 = 1

        newTrainData.X[i][1] = realTime
        newTrainData.X[i][2] = realCity
        newTrainData.X[i][3] = cityType
        newTrainData.X[i][4] = row4
    newTrainData.X = newTrainData.X[1:].astype(np.float)
    newTrainData.y = newTrainData.y[1:].astype(np.float)
    
    newTestData = np.copy(testData)
    for i, row in enumerate(testData):
        if i is 0:
            continue
        # Converts the city into a number
        realCity = 0

        # Converts the time from a string into a number
        month, day, year = ''.join(row[1][0:2]), ''.join(row[1][3:5]), ''.join(row[1][6:10])
        realTime = time.mktime(time.strptime(month + " " + day + " " + year, "%m %d %Y"))

        cityType = 0
        if row[3] == 'Big Cities':
            cityType = 1
            
        row4 = 0
        if row[4] == 'IL':
            row4 = 1

        newTestData[i][1] = realTime
        newTestData[i][2] = realCity
        newTestData[i][3] = cityType
        newTestData[i][4] = row4


    return newTrainData, newTestData[1:].astype(np.float)



