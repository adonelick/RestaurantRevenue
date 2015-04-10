"""
Andrew Donelick
Alex Putman

Machine Learning - CS 158
Final Project: Predicting Restaraunt Revenue

This file is meant to clean up the data we are given.

"""
class Data :
    
    def __init__(self, X, y) :
        """Data class"""
        
        # n = number of examples, d = dimensionality
        self.X = X    # nxd array
        self.y = y    # rank-1 array (think row vector)

def preprocessData(trainData, testData, saveLocation):
    """
    Preprocesses the train and test data files to 
    remove strings, time stamps, and other non-numerical
    values. These preprocessed data files are then saved
    in the location specified, and returned by the function.
    """


    return None, None


if __name__ == '__main__':
    main()


