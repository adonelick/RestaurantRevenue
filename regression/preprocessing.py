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
from pygeocoder import Geocoder

def preprocessData(trainData, testData, saveLocation=None):
    """
    Preprocesses the train and test data files to 
    remove strings, time stamps, and other non-numerical
    values. These preprocessed data files are then saved
    in the location specified, and returned by the function.

    :param trainData: (Data object)
    :param testData: (2-d numpy array)
    :param saveLocation: (string) Location to save new data files to
    :return: (tuple of Data, 2-d numpy array) Preprocessed training/test data
    """

    # Generate a look-up table for all of the possible 
    # cities' lattitude and longitude in the data points.

    cityLocations = {}

    firstRow = True
    for row in trainData.X:
        if firstRow:
            firstRow = False
            continue

        if not cityLocations.has_key(row[2]):
            cityLocations[row[2]] = getLattitudeLongitude(row[2])
            time.sleep(2)

    for row in testData:
        if firstRow:
            firstRow = False
            continue

        if not cityLocations.has_key(row[2]):
            cityLocations[row[2]] = getLattitudeLongitude(row[2])
            time.sleep(2)


    largerX = np.copy(trainData.X)
    largerX = np.insert(largerX, 3, 0, axis=1)
    newTrainData = Data(largerX, np.copy(trainData.y))

    for i, row in enumerate(trainData.X):

        # Skip the first row, which is the CSV file's header
        if i is 0:
            continue

        # Converts the time from a string into a number
        openDate = convertDateToFloat(row[1])

        # Converts the city into a lattitude and longitude
        lattitude, longitude = cityLocations[row[2]]

        cityGroup = 0
        if row[3] == 'Big Cities':
            cityGroup = 1
            
        dataType = 0
        if row[4] == 'IL':
            dataType = 1

        newTrainData.X[i][1] = openDate
        newTrainData.X[i][2] = lattitude
        newTrainData.X[i][3] = longitude
        newTrainData.X[i][4] = cityGroup
        newTrainData.X[i][5] = dataType

    # Convert all other entries from strings to floats
    newTrainData.X = newTrainData.X[1:].astype(np.float)
    newTrainData.y = newTrainData.y[1:].astype(np.float)

    print newTrainData.X.shape
    print newTrainData.y.shape
    

    # Now preprocess the testing (unlabeled) data
    newTestData = np.copy(testData)
    newTestData = np.insert(newTestData, 3, 0, axis=1)
    for i, row in enumerate(testData):

        # Skip the first row, which is the CSV file's header
        if i is 0:
            continue

        # Converts the time from a string into a number
        openDate = convertDateToFloat(row[1])

        # Converts the city into a lattitude and longitude
        lattitude, longitude = cityLocations[row[2]]

        cityGroup = 0
        if row[3] == 'Big Cities':
            cityGroup = 1
            
        dataType = 0
        if row[4] == 'IL':
            dataType = 1

        newTestData[i][1] = openDate
        newTestData[i][2] = lattitude
        newTestData[i][3] = longitude
        newTestData[i][4] = cityGroup
        newTestData[i][5] = dataType

    newTestData = newTestData[1:].astype(np.float)

    return newTrainData, newTestData

def getLattitudeLongitude(city):
    """
    Fetches the lattitude and longitude of a given
    city name.

    :param city: (string) Name of city
    :return: (float tuple) lattitude and longitude of city
    """
    try:
        results = Geocoder.geocode(city)
        lattitude, longitude = results[0].coordinates
    except Exception, e:
        lattitude, longitude = 0, 0
    return lattitude, longitude


def convertDateToFloat(date):
    """
    Converts a date string in the form 
    mm/dd/yyyy to a float.

    :param date: (string) date string to be converted
    :return: (float) seconds since epoch of date
    """
    month = ''.join(date[0:2])
    day = ''.join(date[3:5])
    year = ''.join(date[6:10])
    realTime = time.mktime(time.strptime(month + " " + day + " " + year, "%m %d %Y"))

    return realTime





