import cv2
import numpy

# Precondition for each method: image(s) are in grayscale
# TODO: Gave up on Otsu thresholding. Do background subtraction instead.
class ImgSegmenter():
    def __init__(self, diffThres):
        self.diffThres = diffThres
    # Performs background subtraction to binarize img
    def backgroundSubtract(self, img):
        background = numpy.zeros(img.shape)
        diffImg = numpy.absolute(numpy.subtract(img, background)).astype('uint8')
        _,binImg = cv2.threshold(diffImg, self.diffThres, 255, cv2.THRESH_BINARY)
        return binImg
    
    # Binarizes each image in imgSet using Otsu's method
    def binarizeSet(self, imgSet):
        binImgs = []
        for img in imgSet: # Otsu threshold each image
            self.binarizeImg(img)
        return binImgs
    # Binarizes image using Otsu's method
    def binarizeImg(self, img):
        grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binImg = cv2.threshold(grayImg,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return binImg