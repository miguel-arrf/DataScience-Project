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

# New graph:
# new:

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# data = make_blobs(n_samples=200, n_features=8, centers=6, cluster_std=1.8, random_state=101)

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

from sklearn.cluster import KMeans


var1 = 6
var2 = 7

c = d = 0
for i in range(4):
    ax[c, d].title.set_text(f"{i + 1} iteration points:")
    kmeans = KMeans(n_clusters=6, random_state=0, max_iter=i + 1)
    kmeans.fit(data[["COMPLAINT", "EMOTIONAL_STATUS"]])
    centroids = kmeans.cluster_centers_
    ax[c, d].scatter(data.iloc[:, var1], data.iloc[:, var2], cmap='brg')
    ax[c, d].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='black')
    d += 1
    if d == 2:
        c += 1
        d = 0

show()
