import cv2
import numpy

# Precondition for each method: image(s) are in grayscale
# TODO: Gave up on Otsu thresholding. Do background subtraction instead.
class ImgSegmenter():
    def __init__(self, diffThres):
        self.diffThres = diffThres
    # Binarizes each image in imgSet using Otsu's method
    def binarizeSet(self, imgSet):
        binImgs = []
        for img in imgSet:
            binImg = self.backgroundSubtract(img[0])
            binImgs.append(binImg)
        return binImgs
    # Performs background subtraction to binarize img
    def backgroundSubtract(self, img):
        background = numpy.zeros(img.shape)
        diffImg = numpy.absolute(numpy.subtract(img, background)).astype('uint8')
        _,binImg = cv2.threshold(diffImg, self.diffThres, 255, cv2.THRESH_BINARY)
        binImg = self.removeNoise(binImg)
        return binImg
    # Do additional processing to remove holes in hand, etc
    def removeNoise(self, img):
        kernel = numpy.ones((3,3), numpy.uint8)
        newImg = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        return newImg
