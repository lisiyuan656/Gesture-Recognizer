import neurolab as nl
import numpy as np

class NeuralNet():
    """
    Parameters:
    num_neurons_layer -- List of numbers of neurons in each layer. Number of inputs is first number.
    """
    def __init__(self, num_neurons_layer):
        MAX_FEATURE_VALUE = 100
        minMax = [[0, MAX_FEATURE_VALUE]]*num_neurons_layer[0]
        num_neurons_layer = num_neurons_layer[1:]
        self.net = nl.net.newff(minMax, num_neurons_layer)
    """ Wrapper for Neurolab's train method
    Parameters:
    train_patts -- Training patterns as ndarray
    train_outs -- Expected outputs for training patterns as ndarray
    epochs -- Number of epochs to train
    thres -- Stop training if this threshold is reached
    """
    def train(self, train_patts, train_outs, epochs, thres):
        error = self.net.train_gd(train_patts, train_outs, epochs=epochs, show=100, goal=thres)
        return error
    """ Wrapper for Neurolab's sim method
    Parameters:
    test_patts -- Testing patterns as ndarray
    """
    def predict(self, test_patts):
        output = self.net.sim(test_patts)
        return output