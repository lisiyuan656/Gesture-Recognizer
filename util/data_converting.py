class DataConverter(object):
    ''' Converts data to different formats '''
    def __init__(self, params):
        return
    """ Converts gesture labels to vector form
    Returned vectors have length of 36. All elements are 0 except elements associated with gesture labels,
    which have value of 1. Label to index mapping is as follows:
        Digits 0-9 map to indices 0-9
        Letters a-z map to indices 10-35
    """
    def vectorize_labels(self, labels):
        return