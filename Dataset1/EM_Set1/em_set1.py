import math

from matplotlib import pyplot as plt
from matplotlib.pyplot import subplots, show, savefig
from pandas import DataFrame, read_csv
from sklearn.metrics import silhouette_score, mean_absolute_error
from sklearn.mixture import GaussianMixture

from ds_charts import choose_grid, plot_clusters, plot_line

data: DataFrame = read_csv('../teste_to_use.csv')
y = data["PERSON_INJURY"]
y.replace(('Injured', 'Killed'), (0, 1), inplace=True)

data.pop('PERSON_INJURY')
data.pop('UNIQUE_ID')
print(data.columns)

v1 = 1
v2 = 2

N_CLUSTERS = [2, 3, 5, 7, 9, 11]
rows, cols = choose_grid(len(N_CLUSTERS))

mse: list = []
sc: list = []
mae = []

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


    plot_clusters(data, v2, v1, labels.astype(float), estimator.means_, k,
                  f'EM k={k}', ax=axs[i, j])

    i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)

plt.tight_layout()
savefig(f'images/em_study_before_pca.png',  dpi=300)
show()


fig, ax = subplots(1, 3, figsize=(6, 3), squeeze=False)
plot_line(N_CLUSTERS, mse, title='EM MSE', xlabel='k', ylabel='MSE', ax=ax[0, 0])
plot_line(N_CLUSTERS, mae, title='EM MAE', xlabel='k', ylabel='MAE', ax=ax[0, 2])
plot_line(N_CLUSTERS, sc, title='EM SC', xlabel='k', ylabel='SC', ax=ax[0, 1],percentage=False,
              yLimit=[-1, 1])

plt.tight_layout()
savefig(f'images/em_scores_before_pca.png',  dpi=300)
show()
