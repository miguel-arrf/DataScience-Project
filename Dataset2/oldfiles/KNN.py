from matplotlib.pyplot import show, figure, savefig
from pandas import unique
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from ds_charts import multiple_line_chart, plot_evaluation_results


class KNN:
    def __init__(self, trnX, trnY, tstY, tstX):
        labels = unique(trnY)
        labels.sort()

        nvalues = [ 5, 7, 13, 19]
        dist = ['manhattan', 'euclidean', 'chebyshev']
        values = {}
        best = (0, '')
        last_best = 0
        for d in dist:
            print("d: ", d)
            y_tst_values = []
            for n in nvalues:
                print("\tn: ", n)
                knn = KNeighborsClassifier(n_neighbors=n, metric=d)
                knn.fit(trnX, trnY)
                prd_tst_Y = knn.predict(tstX)
                y_tst_values.append(precision_score(tstY, prd_tst_Y))
                if y_tst_values[-1] > last_best:
                    best = (n, d)
                    last_best = y_tst_values[-1]
            values[d] = y_tst_values

        figure()
        multiple_line_chart(nvalues, values, title='KNN variants', xlabel='n', ylabel="precision",
                            percentage=True)
        savefig(f'images/knn/knn_study.png')
        #show()
        print('Best results with %d neighbors and %s' % (best[0], best[1]))


        clf = knn = KNeighborsClassifier(n_neighbors=best[0], metric=best[1])
        clf.fit(trnX, trnY)
        prd_trn = clf.predict(trnX)
        prd_tst = clf.predict(tstX)
        plot_evaluation_results(["Safe", "Danger"], trnY, prd_trn, tstY, prd_tst, extra="")
        savefig('images/knn/knn_best.png')
        #show()



        # Overfitting

        def plot_overfitting_study(xvalues, prd_trn, prd_tst, name, xlabel, ylabel, extra):
            evals = {'Train': prd_trn, 'Test': prd_tst}
            figure()
            multiple_line_chart(xvalues, evals, ax=None, title=f'Overfitting {name}', xlabel=xlabel, ylabel=ylabel,
                                percentage=True)
            savefig(f'images/knn/overfitting_{extra}.png')

        d = best[1]
        y_tst_values = []
        y_trn_values = []
        for n in nvalues:
            knn = KNeighborsClassifier(n_neighbors=n, metric=d)
            knn.fit(trnX, trnY)
            prd_tst_Y = knn.predict(tstX)
            prd_trn_Y = knn.predict(trnX)
            y_tst_values.append(precision_score(tstY, prd_tst_Y))
            y_trn_values.append(precision_score(trnY, prd_trn_Y))
        plot_overfitting_study(nvalues, y_trn_values, y_tst_values, name=f'KNN_K={d}_precision', xlabel='K',
                               ylabel="precision", extra="precision")

        d = best[1]
        y_tst_values = []
        y_trn_values = []
        for n in nvalues:
            knn = KNeighborsClassifier(n_neighbors=n, metric=d)
            knn.fit(trnX, trnY)
            prd_tst_Y = knn.predict(tstX)
            prd_trn_Y = knn.predict(trnX)
            y_tst_values.append(recall_score(tstY, prd_tst_Y))
            y_trn_values.append(recall_score(trnY, prd_trn_Y))
        plot_overfitting_study(nvalues, y_trn_values, y_tst_values, name=f'KNN_K={d}recall', xlabel='K',
                               ylabel="recall", extra="recall")

        f = open(f"images_withFS/knn/bestModel.txt", "w")
        f.write('Best results with %d neighbors and %s' % (best[0], best[1]))
        f.close()

