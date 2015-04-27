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
import sys
sys.path.append("../regression")

import os
from util import *
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import pylab as P


def main():
    
    print "Loading data from disk"
    labeled_data, unlabeled_data = load_visualization_data()
    normalizedY = (labeled_data.y - np.amin(labeled_data.y)) / (np.amax(labeled_data.y) - np.amin(labeled_data.y))

    print "Plotting Histograms of Labeled Data"


    # First do the labeled data
    for index, column in enumerate(labeled_data.X.T):

        P.figure()
        P.hist(column)
        featureName = columnName(index)

        P.xlabel(featureName)
        P.ylabel("Frequency")
        P.title("Histogram for " + featureName + " in Labeled Data")
        P.savefig("histograms/labeled_data_" + featureName + ".png")
        P.close()

    print "Plotting Histograms of Unlabeled Data"
    # Now, for the unlabeled data (and lots of it)
    for index, column in enumerate(unlabeled_data.T):

        P.figure()
        P.hist(column)
        featureName = columnName(index)

        P.xlabel(featureName)
        P.ylabel("Frequency")
        P.title("Histogram for " + featureName + " in Unlabeled Data")
        P.savefig("histograms/unlabeled_data_" + featureName + ".png")
        P.close()


    print "Plotting Revenue vs. Features"
    for index, column in enumerate(labeled_data.X.T):

        P.figure()
        P.plot(column, labeled_data.y, 'bo')
        featureName = columnName(index)

        P.xlabel(featureName)
        P.ylabel("Revenue")
        P.title("Revenue vs " + featureName + " in Labeled Data")
        P.savefig("revenue/revenue_" + featureName + ".png")
        P.close()

    print "Showing Labeled Data is Unrepresentational of Unlabeled"
    pca = PCA(n_components=2)
    pca.fit(unlabeled_data)
    transformedData = pca.transform(labeled_data.X)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, sample in enumerate(transformedData):

        x = []
        y = []
        z = []

        x.append(sample[0])
        y.append(sample[1])
        z.append(labeled_data.y[i])

        ax.scatter(x, y, z, 'bo')
    
    ax.set_xlabel("PCA Dimension 1")
    ax.set_ylabel("PCA Dimension 2")
    ax.set_zlabel("Revenue")
    plt.title("Revenue vs. PCA Unlabeled PCA Dimensions")
    plt.show()



def columnName(index):
    """
    Retreives a nice name for the given column index. If a nice
    name is not available, it produces 'Feature [index]' 
    as the name.
    """

    if index == 0:
        return "Month"
    if index == 1:
        return "Year"
    if index == 2:
        return "City"
    if index == 3:
        return "City Group"
    if index == 4:
        return "Data Type"

    return "Feature " + str(index)

if __name__ == '__main__':
    main()
