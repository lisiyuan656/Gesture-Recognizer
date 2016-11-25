import random
from img_loading import ImageLoader
from NoiseRemoval import NoiseRemoval
from preprocessing.img_segmenting import ImgSegmenter

data = ImageLoader().getData(5)
data_size = len(data)
data = NoiseRemoval().Gaussian_filter(data, 3)
segmenter = ImgSegmenter()
binData = segmenter.binarizeSet(data)

random.shuffle(data)
train_data_size = 2000
training_data = data[1:train_data_size+1]
testing_data = data[train_data_size+1:]
