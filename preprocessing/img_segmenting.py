import cv2
import numpy

# Precondition for each method: image(s) are in RGB
class ImgSegmenter():
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
    # Normalizes image brightness to remove shadows
    def normalizeImg(self, img):
        hist,bins = numpy.histogram(img.flatten(), 256, [0,256])
        return