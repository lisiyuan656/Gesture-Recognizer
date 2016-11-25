import cv2
import numpy
from scipy import linalg

class PCA(object):
    def __init__(self, basisDim):
        self.basisDim = basisDim
    def getEigVecs(self, img):
        normImg = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        covMat = numpy.cov(normImg)
        evals, evecs = linalg.eigh(covMat)
        sortEvals = numpy.argsort(evals)[::-1]
        evecs = evecs[:,sortEvals]
        evecs = evecs[:, :self.basisDim]
        return evecs