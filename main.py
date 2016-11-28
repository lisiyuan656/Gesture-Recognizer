import random
from img_loading import ImageLoader
from preprocessing.NoiseRemoval import NoiseRemoval
from preprocessing.img_segmenting import ImgSegmenter
from preprocessing.imageScaler import imageScaler
from preprocessing.process_data import process_data

data = ImageLoader().getData(5)
data_size = len(data)
data = imageScaler().scaleDataset(data) # rescale the images
data = NoiseRemoval().Gaussian_filter(data, 3) # blur the images
binData = ImgSegmenter(5).binarizeSet(data) # segment the images

random.shuffle(data)
train_data_size = 2000
training_data = data[1:train_data_size+1]
training_x, training_y = process_data().process_data(training_data)
testing_data = data[train_data_size+1:]



# Calculate eigenvectors
eigenvectors = numpy.zeros((36,15,590))
for i in range(10):
    curchar = chr(i+48)
    temp = numpy.zeros((590,15))
    count = 0
    for tuple in data:
        if tuple[1]==curchar:
            temp = temp + PCA(tuple[0])
            count = count + 1
    if count!=0:
        temp = temp / count
    eigenvectors[i]=temp


for i in range(26):
    curchar = chr(i+97)
    temp = numpy.zeros((590,15))
    count = 0
    for tuple in data:
        if tuple[1]==curchar:
            temp = temp + PCA(tuple[0])
            count = count + 1
    if count!=0:
        temp = temp / count
    eigenvectors[i+10]=temp
