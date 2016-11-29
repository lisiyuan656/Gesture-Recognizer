import numpy as np
import process_img
import process_label

def process_data(dataset, basisDim, mean_eigenvectors):
    train_x = np.array([])
    train_y = np.array([])
    for datapoint in dataset:
        img = datapoint[0]
        label = datapoint[1]
        train_x = np.append(train_x, process_img.process_img(img, basisDim, mean_eigenvectors))
        train_y = np.append(train_y, process_label.process_label(label))
    return train_x, train_y
