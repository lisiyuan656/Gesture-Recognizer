import random
import string
from util.img_loading import ImageLoader
from preprocessing import noise_remover, img_scaler, segmenter
from preprocessing.process_data import process_data
from preprocessing.resize_wrapper import resize_wrapper
from feature import pc_analyzer
# from learning.neural_net_wrapper import NeuralNet
from learning.knn_wrapper import KNNClassifier
import numpy
import pickle
from feature.MomentsCalculator import MomentsCalculator

# Load and preprocess data
data = ImageLoader().getData(5)
data_size = len(data)
#basisDim = 15
data = img_scaler.scaleDataset(data) # rescale the images
#data = resize_wrapper(data,4)
data = noise_remover.Gaussian_filter(data, 3) # blur the images
data = segmenter.binarizeSet(data) # segment the images
# Shuffle data and split into training and test
#random.shuffle(data)
train_data_size = data_size
train_data = data[0:train_data_size]
test_data = data[train_data_size:]
# Initialize features in training set
#category_order = [str(i) for i in range(0,10)] + list(string.ascii_lowercase)
#mean_eigenvectors = pc_analyzer.calculate_mean(train_data, category_order)
#mean_eigenvectors = numpy.asarray(mean_eigenvectors)
train_x = numpy.asarray(MomentsCalculator().Moments(train_data))
# Learn to recognize gestures
knn = KNNClassifier()
knn.train(train_x, [train_data[i][1] for i in range(0,train_data_size)])
output = open('knn15.pkl', 'wb')
pickle.dump(knn, output)
output.close()
#test_x, test_y = process_data(test_data, basisDim, mean_eigenvectors)
#test_x = test_x.reshape(data_size/4,36*basisDim+7+1)
#test_y = [test_data[i][1] for i in range(0,data_size/4)]
#output = knn.predict(test_x)
# input_size = 36*basisDim + 7 + 1
# nn = NeuralNet([input_size,30,36])
# error = nn.train(train_x, train_y, 5, 0.05)
# output = nn.predict([test_data[i,0] for i in range(0,data_size/4)])
