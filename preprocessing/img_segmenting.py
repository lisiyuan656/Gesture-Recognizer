import cv2

class ImgSegmenter():
    # Binarizes each image in imgSet using Otsu's method
    # Precondition: each image in imgSet is in RGB
    def binarizeSet(self, imgSet):
        binImgs = []
        for img in imgSet: # Otsu threshold each image
            self.binarizeImg(img)
        return binImgs
    # Binarizes image using Otsu's method
    # Precondition: img is in RGB
    def binarizeImg(self, img):
        grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binImg = cv2.threshold(grayImg,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return binImg
    # Normalizes image brightness to remove shadows
    def normalizeImg(self, img):
        return