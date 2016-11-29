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

basisDim = 15
""" calculate mean_eigenvectors """
#mean_eigenvectors = calculate_mean(training_data)

training_x, training_y = process_data().process_data(training_data, basisDim, mean_eigenvectors)
testing_data = data[train_data_size+1:]
