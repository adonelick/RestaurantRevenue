"""
Andrew Donelick
Alex Putman

Machine Learning - CS 158
Final Project: Predicting Restaraunt Revenue

This file is meant to clean up the data we are given.

"""
import os
import numpy as np
import util
import time
from pygeocoder import Geocoder
import unicodedata
from sklearn import preprocessing
from sklearn.decomposition import PCA


def preprocessDataForVisualization(trainData, testData):
    """
    Preprocesses the train and test data to remove strings,
    time stamps, and other non-numerical valuess. However,
    unlike the preprocessData function, this function 
    simply maps categorical values to numbers, not categories.
    This function is meant to provide data for visualization.

    :param trainData: (Data object)
    :param testData: (2-d numpy array)
    :return: (tuple of Data, 2-d numpy array) Preprocessed training/test data
    """


    # Generates a mapping from city names to numbers (for plotting)
    cities = {}
    for row in trainData.X:
        if not cities.has_key(row[2]):
            cities[row[2]] = len(cities)

    for row in testData:
        if not cities.has_key(row[2]):
            cities[row[2]] = len(cities)

    largerX = np.copy(trainData.X)
    newTrainData = util.Data(largerX, np.copy(trainData.y))
    
    for i, row in enumerate(trainData.X):

        # Skip the first row, which is the CSV file's header
        if i is 0:
            continue

        # Converts the time from a string into a number
        date = row[1]
        fields = date.split('/')
        month = fields[0]
        day = fields[1]
        year = fields[2]

        cityGroup = 0
        if row[3] == 'Big Cities':
            cityGroup = 1
            
        dataType = 0
        if row[4] == 'IL':
            dataType = 1

        newTrainData.X[i][0] = month
        newTrainData.X[i][1] = year
        newTrainData.X[i][2] = cities[row[2]]
        newTrainData.X[i][3] = cityGroup
        newTrainData.X[i][4] = dataType
    
    
    # Convert all other entries from strings to floats
    newTrainData.X = newTrainData.X[1:].astype(np.float)
    newTrainData.y = newTrainData.y[1:].astype(np.float)


    # Now preprocess the testing (unlabeled) data
    newTestData = np.copy(testData)
    for i, row in enumerate(testData):

        # Skip the first row, which is the CSV file's header
        if i is 0:
            continue

        # Converts the time from a string into a number
        date = row[1]
        fields = date.split('/')
        month = fields[0]
        day = fields[1]
        year = fields[2]

        cityGroup = 0
        if row[3] == 'Big Cities':
            cityGroup = 1
            
        dataType = 0
        if row[4] == 'IL':
            dataType = 1

        newTestData[i][0] = month
        newTestData[i][1] = year
        newTestData[i][2] = cities[row[2]]
        newTestData[i][3] = cityGroup
        newTestData[i][4] = dataType
    
    
    # Convert all other entries from strings to floats
    newTrainData.X = newTrainData.X[1:].astype(np.float)
    newTrainData.y = newTrainData.y[1:].astype(np.float)
    newTestData = newTestData[1:].astype(np.float)

    return newTrainData, newTestData    



def preprocessData(trainData, testData, trainPath=None, testPath=None):
    """
    Preprocesses the train and test data files to 
    remove strings, time stamps, and other non-numerical
    values. These preprocessed data files are then saved
    in the location specified, and returned by the function.

    :param trainData: (Data object)
    :param testData: (2-d numpy array)
    :param trainPath: (string) Location to save new train data file
    :param testPath: (string) Location to save new test data file
    :return: (tuple of Data, 2-d numpy array) Preprocessed training/test data
    """

    # Generate a look-up table for all of the possible 
    # cities' lattitude and longitude in the data points.
    #cityLocations = getCityLocations(trainData, testData)

    largerX = np.copy(trainData.X)
    #largerX = np.insert(largerX, 3, 0, axis=1)
    newTrainData = util.Data(largerX, np.copy(trainData.y))
    
    
    
    for i, row in enumerate(trainData.X):
#
#        # Skip the first row, which is the CSV file's header
        if i is 0:
            continue
#
#        # Converts the time from a string into a number
        #openDate = convertDateToFloat(row[1])/1000
        date = row[1]
        fields = date.split('/')
        month = fields[0]
        day = fields[1]
        year = fields[2]
#        # Converts the city into a lattitude and longitude
#        lattitude, longitude = cityLocations[row[2]]
#
#        cityGroup = 0
#        if row[3] == 'Big Cities':
#            cityGroup = 1
#            
#        dataType = 0
#        if row[4] == 'IL':
#            dataType = 1
#
        newTrainData.X[i][0] = month
        newTrainData.X[i][1] = year
#        newTrainData.X[i][2] = lattitude
#        newTrainData.X[i][3] = longitude
#        newTrainData.X[i][4] = cityGroup
#        newTrainData.X[i][5] = dataType
    
    
    # Convert all other entries from strings to floats
    newTrainData.X = newTrainData.X[1:]#.astype(np.float)
    #newTrainData.X = preprocessing.scale(newTrainData.X)
    newTrainData.y = newTrainData.y[1:]#.astype(np.float)


    # Now preprocess the testing (unlabeled) data
    newTestData = np.copy(testData)
    #newTestData = np.insert(newTestData, 3, 0, axis=1)
    for i, row in enumerate(testData):
#
#        # Skip the first row, which is the CSV file's header
        if i is 0:
            continue
#
#        # Converts the time from a string into a number
        #openDate = convertDateToFloat(row[1])/1000
        date = row[1]
        fields = date.split('/')
        month = fields[0]
        day = fields[1]
        year = fields[2]
#        # Converts the city into a lattitude and longitude
#        lattitude, longitude = cityLocations[row[2]]
#
#        cityGroup = 0
#        if row[3] == 'Big Cities':
#            cityGroup = 1
#        
#        # No idea what this is... 
#        dataType = 0
#        if row[4] == 'IL':
#            dataType = 1
#
        newTestData[i][0] = month
        newTestData[i][1] = year
#        newTestData[i][2] = lattitude
#        newTestData[i][3] = longitude
#        newTestData[i][4] = cityGroup
#        newTestData[i][5] = dataType

    newTestData = newTestData[1:]#.astype(np.float)
    #newTestData = newTestData[:,1:]
    #newTestData = preprocessing.scale(newTestData)
    for col in range(newTestData.shape[1]):
        trainCol = newTrainData.X[:,col]
        testCol = newTestData[:,col]
        le = preprocessing.LabelEncoder()
        le.fit(np.concatenate((trainCol,testCol)))
        newTrainData.X[:,col] = le.transform(trainCol)
        newTestData[:,col] = le.transform(testCol)
  
    newTrainData.X = newTrainData.X.astype(int)
    newTestData = newTestData.astype(int)  
    
    enc = preprocessing.OneHotEncoder()
    enc.fit(np.concatenate((newTrainData.X, newTestData), axis=0))
    
    newTrainData.X = enc.transform(newTrainData.X).toarray()
    newTestData = enc.transform(newTestData).toarray()

    #newTrainData.X = newTrainData.X.astype(np.float)
    #newTestData = newTestData.astype(np.float)  
    # apply PCA
    #norm = preprocessing.Normalizer()
    #norm.fit(np.concatenate((newTrainData.X, newTestData), axis=0))
    #newTrainData.X = norm.transform(newTrainData.X)
    #newTestData = norm.transform(newTestData)
    
    print newTrainData.X
    pca = PCA()
    pca.fit(np.concatenate((newTrainData.X, newTestData), axis=0))
    #newPCATrainData = util.Data(pca.transform(newTrainData.X), newTrainData.y)
    #newPCATestData = pca.transform(newTestData)
    # If desired, save the preprocesed data files
    
    dir = os.path.dirname(util.__file__)
    fTrain = os.path.join(dir, trainPath)
    fTest = os.path.join(dir, testPath)
    if trainPath != None:
        X = newTrainData.X
        y = newTrainData.y
        completeData = np.insert(X, X.shape[1], y, axis=1)
        np.savetxt(fTrain, completeData, delimiter=',')

    if testPath != None:
        np.savetxt(fTest, newTestData, delimiter=',')

    return newTrainData, newTestData

def getCityLocations(trainData, testData):
    """
    Generate a look-up table for all of the possible 
    cities' lattitude and longitude in the data points.

    :param trainData: (Data object)
    :param testData: (2-d numpy array)
    :return: (dict) Mapping of city names to lattitude, longitude tuples
    """

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

    return cityLocations

def getLattitudeLongitude(city):
    """
    Fetches the lattitude and longitude of a given
    city name.

    :param city: (string) Name of city
    :return: (float tuple) lattitude and longitude of city
    """
    city = unicode(city, encoding='utf_8')
    city = unicodedata.normalize('NFKD', city).encode('ascii', 'ignore')
    try:
        results = Geocoder.geocode(city + ", Turkey")
        lattitude, longitude = results[0].coordinates
    except Exception, e:
        print "Error for city: ", city
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





