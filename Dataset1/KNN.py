from matplotlib import pyplot as plt
from matplotlib.pyplot import show, figure, savefig
from pandas import unique
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from ds_charts import multiple_line_chart, plot_evaluation_results


class KNN:
    def __init__(self, trnX, trnY, tstY, tstX, imagePath):
        labels = unique(trnY)
        labels.sort()

        nvalues = [5, 9, 15, 19]
        dist = ['manhattan', 'euclidean', 'chebyshev']
        values = {}
        best = (0, '')
        last_best = 0
        best_model = None
        for d in dist:
            print("d: ", d)
            y_tst_values = []
            for n in nvalues:
                print("\tn: ", n)
                knn = KNeighborsClassifier(n_neighbors=n, metric=d)
                knn.fit(trnX, trnY)
                prd_tst_Y = knn.predict(tstX)
                # y_tst_values.append(precision_score(tstY, prd_tst_Y, pos_label="Killed"))
                y_tst_values.append(recall_score(tstY, prd_tst_Y, pos_label="Killed"))
                if y_tst_values[-1] > last_best:
                    best = (n, d)
                    last_best = y_tst_values[-1]
                    best_model = knn
            values[d] = y_tst_values

        figure()
        multiple_line_chart(nvalues, values, title='KNN variants', xlabel='n_neighbors', ylabel="recall",
                            percentage=True)
        savefig(f'{imagePath}/knn/knn_study.png', dpi=300)
        show()
        print('Best results with %d neighbors and %s' % (best[0], best[1]))

        # clf = knn = KNeighborsClassifier(n_neighbors=best[0], metric=best[1])
        # clf.fit(trnX, trnY)
        prd_trn = best_model.predict(trnX)
        prd_tst = best_model.predict(tstX)
        plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst, extra="KNN")
        savefig(f'{imagePath}/knn/knn_best.png', dpi=300)
        show()

        # Overfitting

        def plot_overfitting_study(xvalues, prd_trn, prd_tst, name, xlabel, ylabel, extra):
            evals = {'Train': prd_trn, 'Test': prd_tst}
            figure()
            multiple_line_chart(xvalues, evals, ax=None, title=f'Overfitting {name}', xlabel=xlabel, ylabel=ylabel,
                                percentage=True)
            plt.tight_layout()
            savefig(f'{imagePath}/knn/overfitting_{extra}.png', dpi=300)


        d = best[1]
        y_tst_values_precision = []
        y_trn_values_precision = []

        y_tst_values_recall = []
        y_trn_values_recall = []
        for n in nvalues:
            knn = KNeighborsClassifier(n_neighbors=n, metric=d)
            knn.fit(trnX, trnY)
            prd_tst_Y = knn.predict(tstX)
            prd_trn_Y = knn.predict(trnX)
            y_tst_values_precision.append(precision_score(tstY, prd_tst_Y, pos_label="Killed"))
            y_trn_values_precision.append(precision_score(trnY, prd_trn_Y, pos_label="Killed"))

            y_tst_values_recall.append(recall_score(tstY, prd_tst_Y, pos_label="Killed"))
            y_trn_values_recall.append(recall_score(trnY, prd_trn_Y, pos_label="Killed"))
        plot_overfitting_study(nvalues, y_trn_values_precision, y_tst_values_precision, name=f'KNN_K={d}_precision',
                               xlabel='K',
                               ylabel="precision", extra="precision")
        plot_overfitting_study(nvalues, y_trn_values_recall, y_tst_values_recall, name=f'KNN_K={d}recall', xlabel='K',
                               ylabel="recall", extra="recall")

        f = open(f"{imagePath}/knn/bestModel.txt", "w")
        f.write('Best results with %d neighbors and %s' % (best[0], best[1]))
        f.close()
