import cv2
import numpy
from scipy import linalg, misc
from matplotlib import pyplot as plt

# Read in grayscale image
test = misc.imread('../data/part1/hand1_0_bot_seg_1_cropped.png')
test = numpy.dot(test[...,:3], [0.299, 0.587, 0.114])
plt.imshow(test, cmap = plt.get_cmap('gray'))
plt.show()
# Get eigenvalues + eigenvectors of covariance matrix
test = cv2.normalize(test.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
covMat = numpy.cov(test)
evals, evecs = linalg.eigh(covMat)
# Sort eigenvalues and get corresponding eigenvectors
sortEvals = numpy.argsort(evals)[::-1]
evecs = evecs[:,sortEvals]
evals = evals[sortEvals]
# Pick n eigenvectors as basis
n = 2
evecs = evecs[:, :n]
newTest = numpy.dot(evecs.T, test).T
newTest = numpy.dot(evecs, newTest.T).T
# Show PCA image
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(newTest[:, 0], newTest[:, 1], '.')
plt.show()