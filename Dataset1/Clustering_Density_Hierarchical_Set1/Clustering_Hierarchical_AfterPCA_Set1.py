import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.linalg import eig
from pandas import DataFrame, read_csv
from matplotlib.pyplot import subplots, show, savefig
from sklearn.decomposition import PCA

from ds_charts import choose_grid, plot_clusters, plot_line, multiple_bar_chart, compute_mse, compute_centroids, \
    get_variable_types, compute_mae, compute_rmse
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

data = read_csv(f'../teste_to_use.csv').sample(frac=0.2)
data = data.dropna()

data.pop('PERSON_INJURY')
data.pop('UNIQUE_ID')

symbolic_vars = get_variable_types(data)['Symbolic']
for symbolic_var in symbolic_vars:
    data[symbolic_var] = pd.factorize(data[symbolic_var])[0]

binary_vars = get_variable_types(data)['Binary']
for binary_var in binary_vars:
    data[binary_var] = pd.factorize(data[binary_var])[0]

mean = (data.mean(axis=0)).tolist()
centered_data = data - mean
cov_mtx = centered_data.cov()
eigvals, eigvecs = eig(cov_mtx)

pca = PCA()
pca.fit(centered_data)
PC = pca.components_
var = pca.explained_variance_
transf = pca.transform(data)

data = pd.concat([pd.Series(transf[:, 0]), pd.Series(transf[:, 1])], axis=1)

v1 = 0
v2 = 1

N_CLUSTERS = [2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
rows, cols = choose_grid(len(N_CLUSTERS))

mse: list = []
mae: list = []
sc: list = []
rows, cols = choose_grid(len(N_CLUSTERS))
_, axs = subplots(rows, cols, figsize=(cols * 5, rows * 5), squeeze=False)
i, j = 0, 0
for n in range(len(N_CLUSTERS)):
    k = N_CLUSTERS[n]
    estimator = AgglomerativeClustering(n_clusters=k)
    estimator.fit(data)
    labels = estimator.labels_
    centers = compute_centroids(data, labels)
    mse.append(compute_mse(data.values, labels, centers))
    mae.append(compute_mae(data.values, labels, centers))
    sc.append(silhouette_score(data, labels))
    plot_clusters(data, v2, v1, labels, centers, k, f'Hierarchical k={k}', ax=axs[i, j])
    i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
plt.tight_layout()
savefig("images/Clustering_Hierarchical_PCA", dpi=300)
show()

fig, ax = subplots(1, 3, figsize=(10, 3), squeeze=False)
plot_line(N_CLUSTERS, mse, title='Hierarchical MSE', xlabel='k', ylabel='MSE', ax=ax[0, 0])
plot_line(N_CLUSTERS, mae, title='Hierarchical MAE', xlabel='k', ylabel='MAE', ax=ax[0, 2])
plot_line(N_CLUSTERS, sc, title='Hierarchical SC', xlabel='k', ylabel='SC', ax=ax[0, 1], percentage=False,
          yLimit=[-1, 1])

plt.tight_layout()
savefig("images/Hierarchical_PCA", dpi=300)
show()

METRICS = ['euclidean', 'cityblock', 'chebyshev', 'cosine', 'jaccard']
LINKS = ['complete', 'average']
k = 3
values_mse = {}
values_rmse = {}
values_mae = {}
values_sc = {}
rows = len(LINKS)
cols = len(METRICS)
_, axs = subplots(rows, cols, figsize=(cols * 5, rows * 5), squeeze=False)
for i in range(len(LINKS)):
    mse: list = []
    rmse: list = []
    mae: list = []
    sc: list = []
    m = LINKS[i]
    print("link: ", m)
    for j in range(len(METRICS)):
        link = METRICS[j]
        print("metric: ", link)
        estimator = AgglomerativeClustering(n_clusters=k, linkage=m, affinity=link)
        estimator.fit(data)
        labels = estimator.labels_
        centers = compute_centroids(data, labels)
        mse.append(compute_mse(data.values, labels, centers))
        rmse.append(compute_rmse(data.values, labels, centers))
        mae.append(compute_mae(data.values, labels, centers))
        sc.append(silhouette_score(data, labels))
        plot_clusters(data, v2, v1, labels, centers, k, f'Hierarchical k={k} metric={m} link={link}', ax=axs[i, j])
    values_mse[m] = mse
    values_rmse[m] = rmse
    values_mae[m] = mse
    values_sc[m] = sc
plt.tight_layout()
savefig("images/MetricsHier_PCA", dpi=300)
show()

values_mse = {}
values_rmse = {}
values_mae = {}
values_sc = {}
for i in range(len(METRICS)):
    mse: list = []
    rmse: list = []
    mae: list = []
    sc: list = []
    m = METRICS[i]
    for j in range(len(LINKS)):
        link = LINKS[j]
        estimator = AgglomerativeClustering(n_clusters=k, linkage=link, affinity=m)
        estimator.fit(data)
        labels = estimator.labels_
        centers = compute_centroids(data, labels)
        mse.append(compute_mse(data.values, labels, centers))
        rmse.append(compute_rmse(data.values, labels, centers))
        mae.append(compute_mae(data.values, labels, centers))
        sc.append(silhouette_score(data, labels))
    values_mse[m] = mse
    values_rmse[m] = rmse
    values_mae[m] = mse
    values_sc[m] = sc

has_neg = False
for metric in values_sc:
    for value in values_sc[metric]:
        if value < 0:
            has_neg = True

_, ax = subplots(1, 3, figsize=(10, 4), squeeze=False)
multiple_bar_chart(LINKS, values_mse, title=f'Hierarchical MSE', xlabel='metric', ylabel='MSE', ax=ax[0, 0])
if has_neg:
    multiple_bar_chart(LINKS, values_sc, title=f'Hierarchical SC', xlabel='metric', ylabel='SC', ax=ax[0, 1],
                       percentage=True)#, newLimitUp=1.0, newLimitDown=-1.0)
else:
    multiple_bar_chart(LINKS, values_sc, title=f'Hierarchical SC', xlabel='metric', ylabel='SC', ax=ax[0, 1],
                       percentage=True)
print("links: ", LINKS)
print("Values_sc: ", values_sc)
multiple_bar_chart(LINKS, values_mae, title=f'Hierarchical MAE', xlabel='metric', ylabel='MAE', ax=ax[0, 2])
plt.tight_layout()
savefig("images/Metrics_Hierarchical_PCA", dpi=300)
show()
