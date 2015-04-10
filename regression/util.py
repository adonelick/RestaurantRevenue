# author      : Jessica Wu
# date        : 01/23/2015
# description : ML utilities

# python libraries
import os

# numpy libraries
import numpy as np

# matplotlib libraries
import matplotlib as mpl
import matplotlib.pyplot as plt

######################################################################
# global settings
######################################################################

mpl.lines.width = 2
mpl.axes.labelsize = 14


######################################################################
# classes
######################################################################

class Data :
    
    def __init__(self) :
        """Data class"""
        
        # n = number of examples, d = dimensionality
        self.X = None    # nxd array
        self.y = None    # rank-1 array (think row vector)
    
    def load(self, filename, labeled=True) :
        """Load csv file into X array of features and y array of labels"""
        
        # determine filename
        import util
        dir = os.path.dirname(util.__file__)
        f = os.path.join(dir, filename)
        
        # load data
        with open(f, 'r') as fid :
            data = np.loadtxt(fid, delimiter=",", dType=None)
        
        # separate features and labels
        if labeled:
            self.X = data[:,:-1]
            self.y = data[:,-1] # rank-1 array (think row vector)
        else:
            self.X = data[:,]
    
    def plot(self) :
        """Plot features and labels"""
        pos = np.nonzero(self.y > 0)  # matlab: find(y > 0)
        neg = np.nonzero(self.y < 0)  # matlab: find(y < 0)
        plt.plot(self.X[pos,0], self.X[pos,1], 'b+', markersize=5)
        plt.plot(self.X[neg,0], self.X[neg,1], 'ro', markersize=5)
        plt.show()

# helper functions
def load_data(filename, labeled=True) :
    """Load csv file into Data class"""
    data = Data()
    data.load(filename, labeled)
    return data