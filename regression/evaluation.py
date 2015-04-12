"""
Andrew Donelick
Alex Putman


Machine Learning - CS 158
Final Project: Predicting Restaraunt Revenue

This file is meant to hold functions related to performance
evaluation, especially for the training set.

"""

import math


def rmse(y_true, y_pred):
    """
    Calculates the root mean squared error (RMSE)
    between the true values and the predicted values.

    :param y_true: (numpy array) true y values
    :param y_pred: (numpy array) predicted y values
    :return: (float) root mean squared error
    """
    n = y_true.shape[0]

    squaredErrors = 0
    for i in xrange(n):
        squaredErrors += (y_true[i] - y_pred[i])**2

    squaredErrors /= (1.0 * n)
    return math.sqrt(squaredErrors)
