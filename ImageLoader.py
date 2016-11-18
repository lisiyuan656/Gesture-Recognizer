from scipy import misc
import os

class ImageLoader():
    def loadImages(self, partNum):
        images = []
        directory = 'data/part' + str(partNum)
        for img in os.listdir(directory):
            image = misc.imread(directory + '/' + img)
            images.append(image)
        return images