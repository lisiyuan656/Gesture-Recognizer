from scipy import ndimage
import cPickle
import os

class ImageLoader():
    def loadImages(self, partNum):
        images = []
        directory = 'data/part' + str(partNum)
        for img in os.listdir(directory):
            image = ndimage.imread(directory + '/' + img, mode='L')
            element = (image, img.split('_')[1])
            images.append(element)
        return images

    def getData(self, num):
        data = []
        for i in range(1, num+1):
            part = self.loadImages(i)
            data = data + part
        return data

    def dumpData(self, data):
        output = open('dataset.pk', 'wb')
        cPickle.dump(data, output)
        output.close()

    def loadData(self):
        input_file = open('dataset.pk', 'rb')
        return cPickle.load(input_file)
