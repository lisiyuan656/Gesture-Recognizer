# import cv2
import numpy
from scipy import ndimage, misc
import matplotlib.pyplot as plt

img = ndimage.imread('../data/part1/hand1_0_bot_seg_1_cropped.png')
hist,bins = numpy.histogram(img.flatten(), 256, [0,256])
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/ cdf.max()
plt.plot(cdf_normalized, color = 'b')
plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.ylim([0,2000])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()
cdf_m = numpy.ma.masked_equal(cdf,0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf = numpy.ma.filled(cdf_m,0).astype('uint8')
img2 = cdf[img]
misc.imsave('normImg.png', img2)