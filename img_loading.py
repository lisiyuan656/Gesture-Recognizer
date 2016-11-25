from scipy import ndimage, misc
import cPickle
import os

class ImageLoader():
    # Reads in images from all parts
    def getData(self, num):
        data = []
        for i in range(1, num+1):
            part = self.loadImages(i)
            data = data + part
        return data
    # Reads in images in data/part#partNum in grayscale
    def loadImages(self, partNum):
        images = []
        directory = 'data/part' + str(partNum)
        for img in os.listdir(directory):
            image = self.loadImage(directory + '/' + img)
            element = (image, img.split('_')[1])
            images.append(element)
        return images
    # Reads in image at path
    def loadImage(self, path):
        image = ndimage.imread(path)
        return image
    
    # CURRENTLY UNUSED    
    def dumpData(self, data):
        output = open('dataset.pk', 'wb')
        cPickle.dump(data, output)
        output.close()

    def loadData(self):
        input_file = open('dataset.pk', 'rb')
        return cPickle.load(input_file)

    def loadAllImage(self):
        images = []
        directory = 'handgesturedataset'
        for img in os.listdir(directory):
            image = misc.imread(directory + '/' + img)
            element = (image, img.split('_')[1])
            images.append(element)
        return images
