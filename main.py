import numpy
import scipy
from ImageLoader import ImageLoader
import random
import NoiseRemoval

imgLoader = ImageLoader()
data = imgLoader.getData(5)
data_size = len(data)
noiseRem = NoiseRemoval()
data = noiseRem.Gaussian_filter(data, 3)
# data resizing
# data segmentation

random.shuffle(data)
training_data_size = 2000
training_data = data[1:train_data_size+1]
testing_data = data[train_data_size+1:]
