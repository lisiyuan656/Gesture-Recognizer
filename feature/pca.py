import cv2
import numpy
from scipy import linalg

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
        means = []
        num = 0 # number of images processed
        for img in train_category:
            eigVects = self.getEigVecs(img)
            if not means:
                means = eigVects
            else:
                for i in range(0,MAX_NUM_EIGVECTS):
                    mean = [(means[i,j]*num+eigVects[i,j])/(num+1) for j in range(0,len(means[i]))]
                    means[i] = numpy.asarray(mean, 'uint8')
            num = num + 1
        self.basisDim = orig_basisDim
        return means
    """ Calculate mean eigenvectors for all gesture categories
    Returns: ndarray with 1st dimension as gesture category indices,
        2nd/3rd dimensions as eigenvectors for each class
    """
    def calculate_mean(self, training_set):
        means = []
        return means