from matplotlib import pyplot as plt
from pandas import DataFrame, read_csv, factorize
import pandas as pd
from matplotlib.pyplot import subplots, savefig, show, figure
from ds_charts import choose_grid, plot_clusters, plot_line, get_variable_types, compute_centroids
from copy import deepcopy
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, mean_absolute_error
from lib.clusteringAlgos import pca, compute_mse, compute_mae
import os, math
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt


def kmeans(data, v1, v2, tag, N_CLUSTERS=[2, 3, 5, 7, 9, 11]):
    rows, cols = choose_grid(len(N_CLUSTERS))
    mse: list = []
    sc: list = []
    mae: list = []

    fig, axs = subplots(rows, cols, figsize=(cols * 5, rows * 5), squeeze=False)
    i = j = 0

    for n in range(len(N_CLUSTERS)):
        print("n: ", N_CLUSTERS[n])
        estimator = KMeans(n_clusters=N_CLUSTERS[n])
        y_pred = estimator.fit_predict(data)
        labels = estimator.labels_
        k = len(set(labels)) - (1 if -1 in labels else 0)

        # axs[i, j].scatter(data.iloc[:, v1], data.iloc[:, v2], c=y_pred)
        # axs[i, j].set_title(f'KMeans k={k}')
        # axs[i, j].set_xlabel("BODILY_INJURY", fontsize=8)
        # axs[i, j].set_ylabel("SAFETY_EQUIPMENT", fontsize=8)

        centers = compute_centroids(data, labels)
        mse.append(compute_mse(data.values, labels, centers))
        mae.append(compute_mae(data.values, labels, centers))
        sc.append(silhouette_score(data, labels))

        # centroids_x = estimator.cluster_centers_[:, 0]
        # centroids_y = estimator.cluster_centers_[:, 1]
        # axs[i, j].scatter(centroids_x, centroids_y, marker="x", s=150, linewidths=5, zorder=10)

        plot_clusters(data, v1, v2, estimator.labels_.astype(float), estimator.cluster_centers_, k, f'KMeans k={k}',
                      ax=axs[i, j])
        i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
    plt.tight_layout()
    savefig(f"images/k-mean-%sPCA-clusters.png" % (tag), dpi=300)
    show()

    fig, ax = subplots(1, 3, figsize=(6, 3), squeeze=False)
    plot_line(N_CLUSTERS, mse, title='K-Means MSE', xlabel='k', ylabel='MSE', ax=ax[0, 0])
    plot_line(N_CLUSTERS, mae, title='K-Means MAE', xlabel='k', ylabel='MAE', ax=ax[0, 2])
    plot_line(N_CLUSTERS, sc, title='K-Means SC', xlabel='k', ylabel='SC', ax=ax[0, 1], percentage=False,
              yLimit=[-1, 1])
    plt.tight_layout()
    # savefig(f"images/k-mean-%sPCA-eval.png" % (tag),  dpi=300)
    show()

    figure(figsize=(6, 3))

    host = host_subplot(111, axes_class=AA.Axes)
    plt.subplots_adjust(right=0.75)

    par1 = host.twinx()
    par2 = host.twinx()

    offset = 30
    new_fixed_axis = par2.get_grid_helper().new_fixed_axis
    par2.axis["right"] = new_fixed_axis(loc="right",
                                        axes=par2,
                                        offset=(0, 0))

    par2.axis["right"].toggle(all=True)

    new_fixed_axis = par1.get_grid_helper().new_fixed_axis
    par1.axis["right"] = new_fixed_axis(loc="right",
                                        axes=par1,
                                        offset=(45, 0))

    par1.axis["right"].toggle(all=True)

    # host.set_xlim(0, 2)
    # host.set_ylim(0, 2)

    host.set_xlabel("k")
    host.set_ylabel("MSE")
    par1.set_ylabel("MAE")
    par2.set_ylabel("SC")

    p1, = host.plot(N_CLUSTERS, mse, label="K-Means MSE")
    p2, = par1.plot(N_CLUSTERS, mae, label="K-Means MAE")
    p3, = par2.plot(N_CLUSTERS, sc, label="K-Means SC")

    # par1.set_ylim(0, 4)
    par2.set_ylim(0, 1)

    host.legend()

    host.axis["left"].label.set_color(p1.get_color())
    par1.axis["right"].label.set_color(p2.get_color())
    par2.axis["right"].label.set_color(p3.get_color())

    plt.draw()
    savefig(f"images/k-mean-%sPCA-eval.png" % (tag), dpi=300)
    plt.show()


if __name__ == "__main__":
    data = read_csv(f'../teste_to_use.csv').sample(frac=0.3)

    data = data.dropna()

    data = data.drop('PERSON_INJURY', 1)
    data = data.drop('UNIQUE_ID', 1)

    for factorizableVar in get_variable_types(data)['Symbolic']:
        data[factorizableVar] = factorize(data[factorizableVar])[0]

    dataBeforePca, dataAfterPca = deepcopy(data), pca(deepcopy(data[["BODILY_INJURY", "SAFETY_EQUIPMENT"]]))
    print(dataBeforePca.columns)
    print(dataBeforePca.columns[1])
    print(dataBeforePca.columns[2])
    kmeans(dataBeforePca, 1, 2, "before")  # we were using 6 and 7
    kmeans(dataAfterPca, 0, 1, "after")
