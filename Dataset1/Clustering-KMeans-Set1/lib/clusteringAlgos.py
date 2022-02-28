from sklearn.decomposition import PCA
from numpy.linalg import eig

import pandas as pd
import math

def pca(data):
    mean = (data.mean(axis=0)).tolist()
    centered_data = data - mean
    cov_mtx = centered_data.cov()
    eigvals, eigvecs = eig(cov_mtx)

    pca = PCA()
    pca.fit(centered_data)
    PC = pca.components_
    var = pca.explained_variance_
    transf = pca.transform(data)

    return pd.concat([pd.Series(transf[:, 0]), pd.Series(transf[:, 1])], axis=1)


def compute_mae(X, labels_in, centroids):
    n_in = len(X)
    centroid_per_record = [centroids[labels_in[a]] for a in range(n_in)]
    partial = X - centroid_per_record
    partial = list(abs(partial))
    partial = [sum(el) for el in partial]
    partial = sum(partial)
    return partial / (n_in - 1)


def compute_mse(X, labels: list, centroids: list) -> float:
    n = len(X)
    centroid_per_record = [centroids[labels[i]] for i in range(n)]
    partial = X - centroid_per_record
    partial = list(partial * partial)
    partial = [sum(el) for el in partial]
    partial = sum(partial)
    return partial / (n-1)


def compute_rmse(X, labels, centroids):
    n_in = len(X)
    centroid_per_record = [centroids[labels[a]] for a in range(n_in)]
    partial = X - centroid_per_record
    partial = list(partial * partial)
    partial = [sum(el) for el in partial]
    partial = sum(partial)
    return math.sqrt(partial / (n_in - 1))