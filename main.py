import random
import string
from util.img_loading import ImageLoader
from preprocessing import noise_remover, img_scaler, segmenter
from preprocessing.process_data import process_data
from feature import pc_analyzer
# from learning.neural_net_wrapper import NeuralNet
from learning.nn_pybrain import nn_pybrain
from learning.knn_wrapper import KNNClassifier
from pybrain.datasets import ClassificationDataSet
import numpy
from pybrain.utilities           import percentError

# Load and preprocess data
data = ImageLoader().getData(1)
data_size = len(data)
basisDim = 15
data = img_scaler.scaleDataset(data) # rescale the images
data = noise_remover.Gaussian_filter(data, 3) # blur the images
binData = segmenter.binarizeSet(data) # segment the images
# Shuffle data and split into training and test
random.shuffle(data)
train_data = numpy.array([])
train_data_size = data_size
image_size = data[0][0].size
for datapoint in data:
    train_data = numpy.append(train_data, datapoint[0].reshape(1, image_size))


print "Finish preprocessing..."
#train_data = data[0:train_data_size]
#test_data = data[train_data_size:]
# Initialize features in training set
category_order = [str(i) for i in range(0,10)] + list(string.ascii_lowercase)
mean_eigenvectors = pc_analyzer.calculate_mean(train_data, category_order)
mean_eigenvectors = numpy.asarray(mean_eigenvectors)
train_x, train_y = process_data(train_data, basisDim, mean_eigenvectors)
train_x = train_x.reshape(train_data_size, 36*basisDim+7+1)
train_x = train_x[:, 36*basisDim:]
train_y = train_y.reshape(train_data_size, 1)
#input_size = 36*basisDim + 7 + 1
input_size = 8
output_size = 36
train_dataset_pybrain = ClassificationDataSet(input_size, nb_classes=36, class_labels=category_order)
train_dataset_pybrain.setField('input', train_x)
train_dataset_pybrain.setField('target', train_y)
tstdata_temp, trndata_temp = train_dataset_pybrain.splitWithProportion(0.25)
print "Finish feature extraction..."
tstdata = ClassificationDataSet(input_size, nb_classes=36, class_labels=category_order)
for n in xrange(0, tstdata_temp.getLength()):
    tstdata.addSample( tstdata_temp.getSample(n)[0], tstdata_temp.getSample(n)[1] )
trndata = ClassificationDataSet(input_size, nb_classes=36, class_labels=category_order)
for n in xrange(0, trndata_temp.getLength()):
    trndata.addSample( trndata_temp.getSample(n)[0], trndata_temp.getSample(n)[1] )
trndata._convertToOneOfMany()
tstdata._convertToOneOfMany()
# Classify using feedforward neural nets
nn = nn_pybrain([input_size, 15, output_size])
nn.enroll(trndata)
print "Begin training neural nets..."
epochs = 200
for i in range(epochs):
    nn.train_for_one_epoch()
    trnresult = percentError(nn.trainer.testOnClassData(), trndata['class'])
    tstresult = percentError(nn.trainer.testOnClassData(dataset=tstdata), tstdata['class'])
    print "epoch: %4d" % nn.trainer.totalepochs, "train error: %5.2f%%" % trnresult, "test error: %5.2f%%" % tstresult

"""
# Learn to recognize gestures
knn = KNNClassifier()
knn.train(train_x, [train_data[i,0] for i in range(0,train_data_size)])
output = knn.predict(test_data)
# nn = NeuralNet([input_size,30,36])
# error = nn.train(train_x, train_y, 5, 0.05)
# output = nn.predict([test_data[i,0] for i in range(0,data_size/4)])
"""
