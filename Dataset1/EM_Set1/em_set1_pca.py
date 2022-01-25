import math

import pandas as pd
from matplotlib.pyplot import subplots, show, savefig
from numpy.linalg import eig
from pandas import DataFrame, read_csv
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, mean_absolute_error
from sklearn.mixture import GaussianMixture

from ds_charts import choose_grid, plot_clusters, plot_line

data: DataFrame = read_csv('../teste_to_use.csv')
y = data["PERSON_INJURY"]
y.replace(('Injured', 'Killed'), (0, 1), inplace=True)

data.pop('PERSON_INJURY')
data.pop('UNIQUE_ID')
print(data.columns)

# start pca

eixo_y = 1
eixo_z = 2

mean = (data.mean(axis=0)).tolist()
centered_data = data - mean
cov_mtx = centered_data.cov()
eigvals, eigvecs = eig(cov_mtx)


pca = PCA()
pca.fit(centered_data)
PC = pca.components_
var = pca.explained_variance_

data = pd.concat([data.iloc[:, eixo_y], data.iloc[:, eixo_z]], axis=1)

# end pca

v1 = 0
v2 = 1


N_CLUSTERS = [5, 9, 13, 17, 19, 21, 27, 29]
rows, cols = choose_grid(len(N_CLUSTERS))

mse: list = []
sc: list = []
mae = []
rmse = []

def compute_mae(X, labels_in, centroids):
    n_in = len(X)
    centroid_per_record = [centroids[labels_in[a]] for a in range(n_in)]
    partial = X - centroid_per_record
    partial = list(abs(partial))
    partial = [sum(el) for el in partial]
    partial = sum(partial)
    return partial / (n_in - 1)


def compute_mse(X, labels, centroids):
    n_in = len(X)
    centroid_per_record = [centroids[labels[a]] for a in range(n_in)]
    partial = X - centroid_per_record
    partial = list(partial * partial)
    partial = [sum(el) for el in partial]
    partial = sum(partial)
    return (partial) / (n_in - 1)


def compute_rmse(X, labels, centroids):
    n_in = len(X)
    centroid_per_record = [centroids[labels[a]] for a in range(n_in)]
    partial = X - centroid_per_record
    partial = list(partial * partial)
    partial = [sum(el) for el in partial]
    partial = sum(partial)
    return math.sqrt(partial / (n_in - 1))


_, axs = subplots(rows, cols, figsize=(cols * 5, rows * 5), squeeze=False)
i, j = 0, 0
for n in range(len(N_CLUSTERS)):
    print("k: ", N_CLUSTERS[n])
    k = N_CLUSTERS[n]
    estimator = GaussianMixture(n_components=k)
    estimator.fit(data)
    labels = estimator.predict(data)
    mse.append(compute_mse(data.values, labels, estimator.means_))
    sc.append(silhouette_score(data, labels))
    mae.append(compute_mae(data.values, labels, estimator.means_))
    rmse.append(compute_rmse(data.values, labels, estimator.means_))


    plot_clusters(data, v2, v1, labels.astype(float), estimator.means_, k,
                  f'EM k={k}', ax=axs[i, j])

    i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)

savefig(f'images/em_study_after_pca.png')
show()


fig, ax = subplots(1, 4, figsize=(6, 3), squeeze=False)
plot_line(N_CLUSTERS, mse, title='EM MSE', xlabel='k', ylabel='MSE', ax=ax[0, 0])
plot_line(N_CLUSTERS, sc, title='EM SC', xlabel='k', ylabel='SC', ax=ax[0, 1], percentage=True)
plot_line(N_CLUSTERS, mae, title='EM MAE', xlabel='k', ylabel='MAE', ax=ax[0, 2])
plot_line(N_CLUSTERS, rmse, title='EM RMSE', xlabel='k', ylabel='RMSE', ax=ax[0, 3])


savefig(f'images/em_scores_after_pca.png')
show()
