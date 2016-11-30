from sklearn.neighbors import KNeighborsClassifier

class KNNClassifier(object):
    ''' Wrapper for sklearn KNN '''
    def __init__(self):
        self.classifier = KNeighborsClassifier()
    """ Wrapper for KNeighborsClassifier.fit() """
    def train(self, data, labels):
        self.classifier.fit(data, labels)
    """ Wrapper for KNeighborsClassifier.predict() """
    def predict(self, test):
        return self.classifier.predict(test)