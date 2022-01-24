from pandas import DataFrame, read_csv, factorize
import pandas as pd
from matplotlib.pyplot import subplots, savefig
from ds_charts import choose_grid, plot_clusters, plot_line, get_variable_types, compute_centroids
from copy import deepcopy
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, mean_absolute_error
from lib.clusteringAlgos import pca, compute_mse, compute_rmse, compute_mae
import os, math


def kmeans(data, v1, v2, tag):
    N_CLUSTERS = [2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]

    rows, cols = choose_grid(len(N_CLUSTERS))
    mse: list = []
    sc: list = []
    mae: list = []
    rmse: list = []
    
    fig, axs = subplots(rows, cols, figsize=(cols*5, rows*5), squeeze=False)
    i = j = 0
    
    for n in range(len(N_CLUSTERS)):
        print("n : ", N_CLUSTERS[n])
        estimator = KMeans(n_clusters=N_CLUSTERS[n])
        estimator.fit(data)
        labels = estimator.labels_
        k = len(set(labels)) - (1 if -1 in labels else 0)

        centers = compute_centroids(data, labels)
        mse.append(compute_mse(data.values, labels, centers))
        rmse.append(compute_rmse(data.values, labels, centers))
        mae.append(compute_mae(data.values, labels, centers))
        sc.append(silhouette_score(data, labels))
    
        
        plot_clusters(data, v2, v1, estimator.labels_.astype(float), estimator.cluster_centers_, k, f'KMeans k={k}', ax=axs[i,j])
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)

    
    savefig(f"images/k-mean-%sPCA-clusters.png" % (tag))

    fig, ax = subplots(2, 2, figsize=(6, 6), squeeze=False)
    plot_line(N_CLUSTERS, mse, title='K-Means MSE', xlabel='k', ylabel='MSE', ax=ax[0, 0])
    plot_line(N_CLUSTERS, rmse, title='K-Means RMSE', xlabel='k', ylabel='RMSE', ax=ax[0, 1])
    plot_line(N_CLUSTERS, mae, title='K-Means MAE', xlabel='k', ylabel='MAE', ax=ax[1, 0])
    plot_line(N_CLUSTERS, sc, title='K-Means SC', xlabel='k', ylabel='SC', ax=ax[1, 1], percentage=True)
    savefig(f"images/k-mean-%sPCA-eval.png" % (tag))


if __name__ == "__main__":
    data = read_csv(f'../../data/air_quality_tabular.csv').sample(frac=0.1)
    data = data.dropna()

    data = data.drop('ALARM', 1)
    data = data.drop('FID', 1)
    
    for factorizableVar in get_variable_types(data)['Symbolic']:
        data[factorizableVar] = factorize(data[factorizableVar])[0]
    
    dataBeforePca, dataAfterPca = deepcopy(data), pca(deepcopy(data))
    
    kmeans(dataBeforePca, 18, 22, "before")
    kmeans(dataAfterPca, 0, 1, "after")
    