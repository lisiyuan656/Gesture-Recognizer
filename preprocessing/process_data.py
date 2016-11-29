import numpy as np
import process_img
import process_label

def process_data(dataset, basisDim, mean_eigenvectors):
    training_x = np.array([])
    training_y = np.array([])
    for datapoint in dataset:
        img = datapoint[0]
        label = datapoint[1]
        training_x = np.append(training_x, process_img(img, basisDim, mean_eigenvectors))
        training_y = np.append(training_y, process_label(label))
    return training_x, training_y
