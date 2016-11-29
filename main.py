import random
import string
from img_loading import ImageLoader
from preprocessing import noise_remover, img_scaler, segmenter
from preprocessing.process_data import process_data
from feature import pc_analyzer

# Load and preprocess data
data = ImageLoader().getData(5)
data_size = len(data)
data = img_scaler.scaleDataset(data) # rescale the images
data = noise_remover.Gaussian_filter(data, 3) # blur the images
binData = segmenter.binarizeSet(data) # segment the images
# Shuffle data and split into training and test
random.shuffle(data)
train_data_size = 2000
train_data = data[0:train_data_size]
testing_data = data[train_data_size:]
# Initialize features in training set
category_order = [str(i) for i in range(0,10)] + list(string.ascii_lowercase)
mean_eigenvectors = pc_analyzer.calculate_mean(train_data, category_order)
train_x, train_y = process_data().process_data(train_data, mean_eigenvectors)
