from scipy import misc
import os

class ImageLoader():
    def loadImages(self, partNum):
        images = []
        directory = 'data/part' + str(partNum)
        for img in os.listdir(directory):
            image = misc.imread(directory + '/' + img)
            element = (image, img.split('_')[1])
            images.append(element)
        return images

    def getData(self, num):
        data = []
        for i in range(num):
            part = self.loadImages(i)
            data = data + part
        return data

    def loadAllImage(self):
        images = []
        directory = 'handgesturedataset'
        for img in os.listdir(directory):
            image = misc.imread(directory + '/' + img)
            element = (image, img.split('_')[1])
            images.append(element)
        return images
