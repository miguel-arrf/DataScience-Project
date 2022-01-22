import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import subplots, show, savefig
from pandas import DataFrame, read_csv
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, pairwise_distances_argmin

from ds_charts import choose_grid, plot_clusters, plot_line

data: DataFrame = read_csv('../teste_to_use.csv')
y = data["PERSON_INJURY"]
y.replace(('Injured', 'Killed'), (0, 1), inplace=True)

data.pop('PERSON_INJURY')
data.pop('UNIQUE_ID')
print(data.columns)

v1 = 6
v2 = 7

print(data.columns[6])
print(data.columns[7])

N_CLUSTERS = [2, 3, 4, 5, 6, 7]
rows, cols = choose_grid(len(N_CLUSTERS))

mse: list = []
sc: list = []
davies_bouldin_score_list = []

# data = data[["COMPLAINT", "EMOTIONAL_STATUS"]]

fig, axs = subplots(rows, cols, figsize=(cols * 5, rows * 5), squeeze=False)
i, j = 0, 0
for n in range(len(N_CLUSTERS)):
    k = N_CLUSTERS[n]
    print("k: ", k)
    estimator = KMeans(n_clusters=k)
    estimator.fit(data[["COMPLAINT", "EMOTIONAL_STATUS"]])
    mse.append(estimator.inertia_)

    # mae.append(math.sqrt(estimator.inertia_))
    sc.append(silhouette_score(data[["COMPLAINT", "EMOTIONAL_STATUS"]], estimator.labels_))
    labels = estimator.predict(data[["COMPLAINT", "EMOTIONAL_STATUS"]])

    import config as cfg

    davies_bouldin_score_list.append(davies_bouldin_score(data[["COMPLAINT", "EMOTIONAL_STATUS"]], estimator.labels_))

    centroids = estimator.cluster_centers_
    axs[i, j].scatter(data.iloc[:, v1], data.iloc[:, v2], c=estimator.labels_.astype(float), cmap=cfg.cmap_active)
    # axs[i, j].scatter(estimator.cluster_centers_[:, 0], estimator.cluster_centers_[:, 1], s=200, c='black')
    centers = estimator.cluster_centers_
    for a, col in zip(range(k), cfg.ACTIVE_COLORS):
        cluster_center = centers[a]
        axs[i, j].plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k',
                       markersize=6)
    # plot_clusters(data, v2, v1, estimator.labels_.astype(float), estimator.cluster_centers_, k, f'KMeans k={k}',
    #              ax=axs[i, j])

    i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
savefig(f'images/kmeans_study_before_pca.png')
show()

fig, ax = subplots(1, 3, figsize=(6, 3), squeeze=False)
plot_line(N_CLUSTERS, mse, title='KMeans MSE', xlabel='k', ylabel='MSE', ax=ax[0, 0])
plot_line(N_CLUSTERS, sc, title='KMeans SC', xlabel='k', ylabel='SC', ax=ax[0, 1], percentage=True)
plot_line(N_CLUSTERS, davies_bouldin_score_list, title='KMeans davies_bouldin_score', xlabel='k',
          ylabel='davies_bouldin_score', ax=ax[0, 2], percentage=True)

savefig(f'images/kmeans_scores_before_pca.png')
show()
