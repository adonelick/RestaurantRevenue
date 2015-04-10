"""
Andrew Donelick
Alex Putman

Machine Learning - CS 158
Final Project: Predicting Restaraunt Revenue

This file is meant to clean up the data we are given.

"""

import numpy as np

class Data :
    
    def __init__(self, X, y) :
        """Data class"""
        
        # n = number of examples, d = dimensionality
        self.X = X    # nxd array
        self.y = y    # rank-1 array (think row vector)

    def setX(self, X):
        self.X = X

    def set_y(self, y):
        self.y = y

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
        for row in trainData.X:

            # Converts the city into a number
            realCity = 0

            # Converts the time from a string into a number
            month, day, year = ''.join(row[1][0:2]), ''.join(row[1][3:5]), ''.join(row[1][6:10])
            realTime = time.mktime(time.strptime(month + " " + day + " " + year, "%m %d %Y"))

            cityType = 0
            if row[3] == 'Big Cities':
                cityType = 1

            newTrainData.X[1] = realTime
            newTrainData.X[2] = realCity
            newTrainData.X[3] = cityType

        newTestData = np.copy(testData)
        for row in testData.X:

            # Converts the city into a number
            realCity = 0

            # Converts the time from a string into a number
            month, day, year = ''.join(row[1][0:2]), ''.join(row[1][3:5]), ''.join(row[1][6:10])
            realTime = time.mktime(time.strptime(month + " " + day + " " + year, "%m %d %Y"))

            cityType = 0
            if row[3] == 'Big Cities':
                cityType = 1

            newTestData[1] = realTime
            newTestData[2] = realCity
            newTestData[3] = cityType


    return newTrainData, newTestData


if __name__ == '__main__':
    main()


