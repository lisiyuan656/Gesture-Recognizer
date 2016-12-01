import numpy as np
import sys
sys.path.append("..")
from feature.pca import PCA
from feature.MomentsCalculator import MomentsCalculator
from feature.Interests import Interest_points

def process_img(img, basisDim, data_mean_evecs):
    res = np.array([])
    num_minmax = 5
    eigenvecs = PCA(basisDim).getEigVecs(img)
    for i in range(basisDim):
        for j in range(36):
            dis = np.linalg.norm(eigenvecs[:,i]-data_mean_evecs[j,:,i])
            res = np.append(res, dis)
    res = np.append(res, np.asarray(MomentsCalculator().ImageMoments(img)))
    #res = np.append(res, Interest_points().get_min_max_distance(img, num_minmax))
    res = np.append(res, Interest_points().fast_feature(img))
    return res
