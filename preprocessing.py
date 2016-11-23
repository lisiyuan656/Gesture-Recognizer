import cv2

class ImgSegmenter():
    # Binarizes each image in imgSet using Otsu's method
    def binarizeSet(self, imgSet):
        binImgs = []
        for img in imgSet: # Otsu threshold each image
            self.binarizeImg(img)
        return binImgs
    # Binarizes image using Otsu's method
    def binarizeImg(self, img):
        _, binImg = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return binImg
    # def dilateImg(self, img):