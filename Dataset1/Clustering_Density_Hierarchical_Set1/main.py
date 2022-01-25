from pandas import DataFrame, read_csv, factorize
from matplotlib.pyplot import subplots, savefig
from copy import deepcopy
from ds_charts import choose_grid, plot_clusters, plot_line, bar_chart, compute_mse, compute_centroids, get_variable_types
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from lib.clusteringAlgos import pca, compute_mae, compute_rmse
import numpy as np
from scipy.spatial.distance import pdist, squareform

def EPSClustering(data, v1, v2, tag, N_CLUSTERS = [2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]):
    EPS = [2.5, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    rows, cols = choose_grid(len(EPS))
    mse: list = []
    sc: list = []
    mae: list = []
    rmse: list = []


    rows, cols = choose_grid(len(EPS))
    _, axs = subplots(rows, cols, figsize=(cols*5, rows*5), squeeze=False)
    i, j = 0, 0
    for n in range(len(EPS)):
        print("n: ", EPS[n])
        estimator = DBSCAN(eps=EPS[n], min_samples=2)
        estimator.fit(data)
        labels = estimator.labels_
        k = len(set(labels)) - (1 if -1 in labels else 0)
        if k > 1:
            centers = compute_centroids(data, labels)
            
            mse.append(compute_mse(data.values, labels, centers))
            rmse.append(compute_rmse(data.values, labels, centers))
            mae.append(compute_mae(data.values, labels, centers))
            sc.append(silhouette_score(data, labels))
            
            plot_clusters(data, v2, v1, labels.astype(float), estimator.components_, k, f'DBSCAN eps={EPS[n]} k={k}', ax=axs[i,j])
            i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
        else:
            mse.append(0)
            rmse.append(0)
            mae.append(0)
            sc.append(0)
    
    savefig(f"images/EPS_Clustering - %s" % (tag))


    fig, ax = subplots(2, 2, figsize=(6, 6), squeeze=False)
    plot_line(EPS, mse, title='DBSCAN MSE', xlabel='eps', ylabel='MSE', ax=ax[0, 0])
    plot_line(EPS, rmse, title='DBSCAN RMSE', xlabel='eps', ylabel='RMSE', ax=ax[0, 1])
    plot_line(EPS, mae, title='DBSCAN MAE', xlabel='eps', ylabel='MAE', ax=ax[1, 0])
    plot_line(EPS, sc, title='DBSCAN SC', xlabel='eps', ylabel='SC', ax=ax[1, 1], percentage=True)
    savefig(f"images/EPS_DBSCAN-%s" % (tag))




def metrics(data, v1, v2, tag):
    METRICS = ['euclidean', 'cityblock', 'chebyshev', 'cosine', 'jaccard']
    distances = []
    for m in METRICS:
        dist = np.mean(np.mean(squareform(pdist(data.values, metric=m))))
        distances.append(dist)

    print('%s - AVG distances among records' % (tag), distances)
    distances[0] *= 0.6
    distances[1] = 80
    distances[2] *= 0.6
    distances[3] *= 0.1
    distances[4] *= 0.15
    print('%s - CHOSEN EPS' % (tag), distances)

    rmse: list = []
    mae: list = []
    mse: list = []
    sc: list = []
    rows, cols = choose_grid(len(METRICS))
    _, axs = subplots(rows, cols, figsize=(cols*5, rows*5), squeeze=False)
    i, j = 0, 0
    for n in range(len(METRICS)):
        estimator = DBSCAN(eps=distances[n], min_samples=2, metric=METRICS[n])
        estimator.fit(data)
        labels = estimator.labels_
        k = len(set(labels)) - (1 if -1 in labels else 0)
        if k > 1:
            centers = compute_centroids(data, labels)
            mse.append(compute_mse(data.values, labels, centers))
            mae.append(compute_mae(data.values, labels, centers))
            rmse.append(compute_rmse(data.values, labels, centers))
            sc.append(silhouette_score(data, labels))
            plot_clusters(data, v2, v1, labels.astype(float), estimator.components_, k, f'DBSCAN metric={METRICS[n]} eps={distances[n]:.2f} k={k}', ax=axs[i,j])
        else:
            mse.append(0)
            rmse.append(0)
            mae.append(0)
            sc.append(0)
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
    
    savefig(f"images/DBSCAN-Metrics-Density - %s" % (tag))

    fig, ax = subplots(2, 2, figsize=(20, 20), squeeze=False)
    bar_chart(METRICS, mse, title='DBSCAN MSE', xlabel='metric', ylabel='MSE', ax=ax[0, 0])
    bar_chart(METRICS, rmse, title='DBSCAN RMSE', xlabel='metric', ylabel='RMSE', ax=ax[0, 1])
    bar_chart(METRICS, mae, title='DBSCAN MAE', xlabel='metric', ylabel='MAE', ax=ax[1, 0])
    bar_chart(METRICS, sc, title='DBSCAN SC', xlabel='metric', ylabel='SC', ax=ax[1, 1], percentage=True)
    savefig(f"images/Metrics_DBSCAN - %s" % (tag))

if __name__ == "__main__":
    data = read_csv(f'../teste_to_use.csv').sample(frac=0.2)
    data = data.dropna()
    y = data["PERSON_INJURY"]
    y.replace(('Injured', 'Killed'), (0, 1), inplace=True)

    data.pop('PERSON_INJURY')
    data.pop('UNIQUE_ID')
    
    for factorizableVar in get_variable_types(data)['Symbolic']:
        data[factorizableVar] = factorize(data[factorizableVar])[0]
    
    dataBeforePca, dataAfterPca = deepcopy(data), pca(deepcopy(data))
    
    EPSClustering(dataBeforePca, 1, 2, "beforePCA")
    metrics(dataBeforePca, 1, 2, "beforePCA")

    EPSClustering(dataAfterPca, 0, 1, "afterPCA")
    metrics(dataAfterPca, 0, 1, "afterPCA")