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

from util import *
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D




def main():
    
    labeled_data, unlabeled_data = load_all_data()
    pca = PCA(n_components=3)
    pca.fit(unlabeled_data[:,1:])
    transformedData = pca.transform(unlabeled_data[:,1:][0:1000])
    
    x = []
    y = []
    z = []

    for sample in transformedData:
        x.append(sample[0])
        y.append(sample[1])
        z.append(sample[2])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, c='b')
    plt.show()






if __name__ == '__main__':
    main()
