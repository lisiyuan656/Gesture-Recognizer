from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import LinearLayer, SigmoidLayer, SoftmaxLayer
from pybrain.supervised.trainers import BackpropTrainer

class nn_pybrain():
    def __init__(self, num_neurons_layer):
        """ num_neurons_layer is the number of neurons of each layer """
        self.net = buildNetwork(*num_neurons_layer, outclass=SoftmaxLayer)
        self.trainer = 0

    def enroll(self, data):
        self.trainer = BackpropTrainer(self.net, data)

    def train_for_one_epoch(self):
        self.trainer.train()

    def train(self, epochs):
        for i in range(epochs):
            error = self.train_for_one_epoch()
            print 'Epoch {0} finished'.format(i)
