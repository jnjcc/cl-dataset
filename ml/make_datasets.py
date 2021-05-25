# -*- coding: utf-8 -*-

import numpy as np
from sklearn.datasets import make_blobs

### iris dataset

### california housing

### gaussian blob for k-Means
def blob_kmeans(fpath):
    blob_centers = np.array(
        [[ 0.2, 2.3],
         [-1.5, 2.3],
         [-2.8, 1.8],
         [-2.8, 2.8],
         [-2.8, 1.3]])
    blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])
    X, y = make_blobs(n_samples = 2000, centers = blob_centers,
            cluster_std = blob_std, random_state = 7)
    y = y.reshape(y.shape[0], 1)
    y = y.astype(int)
    array = np.concatenate((X, y), axis = 1)
    np.savetxt(fpath, array, delimiter = ',', fmt = "%.6f,%.6f,%d")

blob_kmeans("./sklearn/kmeans.blob")

### gaussian blob for GMM
def blob_gmm(fpath):
    X1, y1 = make_blobs(n_samples = 1000, centers = ((4, -4), (0, 0)), random_state = 42)
    X1 = X1.dot(np.array([[0.374, 0.95], [0.732, 0.598]]))
    X2, y2 = make_blobs(n_samples = 250, centers = 1, random_state = 42)
    X2 = X2 + [6, -8]
    X = np.r_[X1, X2]
    y = np.r_[y1, y2]
    y = y.reshape(y.shape[0], 1)
    array = np.concatenate((X, y), axis = 1)
    np.savetxt(fpath, array, delimiter = ',', fmt = "%.6f,%.6f,%d")

blob_gmm("./sklearn/gmm.blob")
