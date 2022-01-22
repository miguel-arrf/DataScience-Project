import pandas as pd
from pandas import DataFrame, read_csv
from matplotlib.pyplot import subplots, show, savefig
from ds_charts import choose_grid, plot_clusters, plot_line, multiple_bar_chart, compute_mse, compute_centroids, \
    get_variable_types, compute_mae, compute_rmse
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

target = 'ALARM'
data = read_csv(f'../../data/air_quality_tabular.csv')
data = data.dropna()
data = data.sample(20000)
symbolic_vars = get_variable_types(data)['Symbolic']
for symbolic_var in symbolic_vars:
    data[symbolic_var] = pd.factorize(data[symbolic_var])[0]

binary_vars = get_variable_types(data)['Binary']
for binary_var in binary_vars:
    data[binary_var] = pd.factorize(data[binary_var])[0]
y = data['ALARM']
data = data.drop('ALARM', 1)
data = data.drop('FID', 1)

v1 = 18
v2 = 22

N_CLUSTERS = [2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
rows, cols = choose_grid(len(N_CLUSTERS))

mse: list = []
rmse: list = []
mae: list = []
sc: list = []
rows, cols = choose_grid(len(N_CLUSTERS))
_, axs = subplots(rows, cols, figsize=(cols*5, rows*5), squeeze=False)
i, j = 0, 0
for n in range(len(N_CLUSTERS)):
    k = N_CLUSTERS[n]
    estimator = AgglomerativeClustering(n_clusters=k)
    estimator.fit(data)
    labels = estimator.labels_
    centers = compute_centroids(data, labels)
    mse.append(compute_mse(data.values, labels, centers))
    rmse.append(compute_rmse(data.values, labels, centers))
    mae.append(compute_mae(data.values, labels, centers))
    sc.append(silhouette_score(data, labels))
    plot_clusters(data, v2, v1, labels, centers, k, f'Hierarchical k={k}', ax=axs[i,j])
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
savefig("images/Clustering_Hierarchical")
#show()

fig, ax = subplots(2, 2, figsize=(6, 6), squeeze=False)
plot_line(N_CLUSTERS, mse, title='Hierarchical MSE', xlabel='k', ylabel='MSE', ax=ax[0, 0])
plot_line(N_CLUSTERS, rmse, title='Hierarchical RMSE', xlabel='k', ylabel='RMSE', ax=ax[0, 1])
plot_line(N_CLUSTERS, mae, title='Hierarchical MAE', xlabel='k', ylabel='MAE', ax=ax[1, 0])
plot_line(N_CLUSTERS, sc, title='Hierarchical SC', xlabel='k', ylabel='SC', ax=ax[1, 1], percentage=True)
savefig("images/Hierarchical")
#show()

METRICS = ['euclidean', 'cityblock', 'chebyshev', 'cosine', 'jaccard']
LINKS = ['complete', 'average']
k = 3
values_mse = {}
values_rmse = {}
values_mae = {}
values_sc = {}
rows = len(METRICS)
cols = len(LINKS)
_, axs = subplots(rows, cols, figsize=(cols*5, rows*5), squeeze=False)
for i in range(len(METRICS)):
    mse: list = []
    rmse : list = []
    mae: list = []
    sc: list = []
    m = METRICS[i]
    for j in range(len(LINKS)):
        link = LINKS[j]
        estimator = AgglomerativeClustering(n_clusters=k, linkage=link, affinity=m )
        estimator.fit(data)
        labels = estimator.labels_
        centers = compute_centroids(data, labels)
        mse.append(compute_mse(data.values, labels, centers))
        rmse.append(compute_rmse(data.values, labels, centers))
        mae.append(compute_mae(data.values, labels, centers))
        sc.append(silhouette_score(data, labels))
        plot_clusters(data, v2, v1, labels, centers, k, f'Hierarchical k={k} metric={m} link={link}', ax=axs[i,j])
    values_mse[m] = mse
    values_rmse[m] = rmse
    values_mae[m] = mse
    values_sc[m] = sc
savefig("images/MetricsHier")
#show()

_, ax = subplots(2, 2, figsize=(6, 6), squeeze=False)
multiple_bar_chart(LINKS, values_mse, title=f'Hierarchical MSE', xlabel='metric', ylabel='MSE', ax=ax[0, 0])
multiple_bar_chart(LINKS, values_rmse, title=f'Hierarchical RMSE', xlabel='metric', ylabel='RMSE', ax=ax[0, 1])
multiple_bar_chart(LINKS, values_mae, title=f'Hierarchical MAE', xlabel='metric', ylabel='MAE', ax=ax[1, 0])
multiple_bar_chart(LINKS, values_sc, title=f'Hierarchical SC', xlabel='metric', ylabel='SC', ax=ax[1, 1], percentage=True)
savefig("images/Metrics_Hierarchical")
#show()