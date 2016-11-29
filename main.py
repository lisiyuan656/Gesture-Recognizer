import random
from img_loading import ImageLoader
from preprocessing import noise_remover, img_scaler, segmenter
from preprocessing.process_data import process_data
import numpy as np
from feature import pc_analyzer

data = ImageLoader().getData(5)
data_size = len(data)
data = img_scaler.scaleDataset(data) # rescale the images
data = noise_remover.Gaussian_filter(data, 3) # blur the images
binData = segmenter.binarizeSet(data) # segment the images

random.shuffle(data)
train_data_size = 2000
training_data = data[1:train_data_size+1]


# Calculate eigenvectors
# mean_eigenvectors = np.zeros((36,15,590))
# mean_eigenvectors = calculate_mean(training_data)
""" input is the training_data, formated the same as data
    output should be a ndarray of dim (36, 590, basisDim)
    code below doesn't work, but should be helpful
"""
"""
for i in range(10):
    curchar = chr(i+48)
    temp = np.zeros((590,15))
    count = 0
    for tuple in training_data:
        if tuple[1]==curchar:
            temp = temp + PCA(tuple[0])
            count = count + 1
    if count!=0:
        temp = temp / count
    eigenvectors[i]=temp
for i in range(26):
    curchar = chr(i+97)
    temp = np.zeros((590,15))
    count = 0
    for tuple in training_data:
        if tuple[1]==curchar:
            temp = temp + PCA(tuple[0])
            count = count + 1
    if count!=0:
        temp = temp / count
    eigenvectors[i+10]=temp
"""
training_x, training_y = process_data().process_data(training_data)
testing_data = data[train_data_size+1:]
