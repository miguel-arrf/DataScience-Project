import math

from numpy import ndarray
from pandas import DataFrame, read_csv
from matplotlib.pyplot import subplots, show, savefig
from ds_charts import choose_grid, plot_clusters, plot_line
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, mean_squared_error, mean_absolute_error, davies_bouldin_score
from sklearn import metrics

data: DataFrame = read_csv('../teste_to_use.csv')
y = data["PERSON_INJURY"]
y.replace(('Injured', 'Killed'), (0, 1), inplace=True)

data.pop('PERSON_INJURY')
data.pop('UNIQUE_ID')
print(data.columns)

v1 = 1
v2 = 2

N_CLUSTERS = [5, 9, 13, 17, 19, 21, 27, 29]
rows, cols = choose_grid(len(N_CLUSTERS))

mse: list = []
sc: list = []
davies_bouldin_score_list = []

fig, axs = subplots(rows, cols, figsize=(cols * 5, rows * 5), squeeze=False)
i, j = 0, 0
for n in range(len(N_CLUSTERS)):
    k = N_CLUSTERS[n]
    print("k: ", k)
    estimator = KMeans(n_clusters=k)
    estimator.fit(data)
    mse.append(estimator.inertia_)

    #mae.append(math.sqrt(estimator.inertia_))
    sc.append(silhouette_score(data, estimator.labels_))
    labels = estimator.predict(data)

    davies_bouldin_score_list.append(davies_bouldin_score(data, estimator.labels_))

    plot_clusters(data, v2, v1, estimator.labels_.astype(float), estimator.cluster_centers_, k, f'KMeans k={k}',
                  ax=axs[i, j])

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
