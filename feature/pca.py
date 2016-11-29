import cv2
import numpy
from scipy import linalg
from __builtin__ import int

# Precondition for all methods: passed image is in grayscale
class PCA(object):
    def __init__(self, basisDim):
        self.basisDim = basisDim
    """ Get eigenvectors associated with n largest eigenvalues, where n is basis dimension """
    def getEigVecs(self, img):
        normImg = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        covMat = numpy.cov(normImg)
        evals, evecs = linalg.eigh(covMat)
        sortEvals = numpy.argsort(evals)[::-1]
        evecs = evecs[:,sortEvals]
        evecs = evecs[:, :self.basisDim]
        return evecs
    """ Calculate mean eigenvectors for training set (assumed to all be of same gesture category) """
    def calcMeanEigvects(self, train_category):
        MAX_NUM_EIGVECTS = 15
        orig_basisDim = self.basisDim # original basis dimension
        self.basisDim = MAX_NUM_EIGVECTS
        means = -1
        num = 0 # number of images processed
        for img in train_category:
            eigVects = self.getEigVecs(img)
            if num == 0:
                means = eigVects
            else:
                for i in range(0,MAX_NUM_EIGVECTS):
                    mean = [(means[i,j]*num+eigVects[i,j])/(num+1) for j in range(0,len(means[i]))]
                    means[i] = numpy.asarray(mean, 'uint8')
            num = num + 1
        self.basisDim = orig_basisDim
        return means
    """ Calculate mean eigenvectors of training set for all gesture categories
    Assumes that data elements are tuples of image and label
    Parameters:
    training_set -- training set for which to calculate mean eigenvectors
    category_order -- order of categories for returned list
    Returns: list of ndarrays of eigenvectors for each class
    """
    def calculate_mean(self, training_set, category_order):
        means = []
        category_sets = self.categorizePatterns(training_set)
        for category in category_order:
            category_means = self.calcMeanEigvects(category_sets[category])
            means.append(category_means)
        return means
    """ Sorts training patterns into categories
    Assumes that patterns are tuples of image and label
    """
    def categorizePatterns(self, patterns):
        category_sets = {} # map of category label to category set
        for pattern in patterns:
            if pattern[1] in category_sets:
                category_sets[pattern[1]].append(pattern[0])
            else:
                category_sets[pattern[1]] = [pattern[0]]
        return category_sets