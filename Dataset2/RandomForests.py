from matplotlib import pyplot as plt
from matplotlib.pyplot import show, figure, savefig, subplots
from pandas import unique
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

from ds_charts import multiple_line_chart, horizontal_bar_chart, plot_evaluation_results, HEIGHT


class RandomForests:
    def __init__(self, trnX, trnY, tstY, tstX):
        labels = unique(trnY)
        labels.sort()

        n_estimators = [5, 10, 25, 50, 100, 400, 1000]
        max_depths = [5, 10, 25]
        max_features = [.3, .5, .7, 1]
        best = ('', 0, 0)
        last_best = 0
        best_model = None


        cols = len(max_depths)
        figure()
        fig, axs = subplots(1, cols, figsize=(cols * HEIGHT, HEIGHT), squeeze=False)
        for k in range(len(max_depths)):
            d = max_depths[k]
            print("k: ", d)
            values = {}
            for f in max_features:
                print("\tf: ", f)
                yvalues = []
                for n in n_estimators:
                    print("\t\tn: ", n)
                    rf = RandomForestClassifier(n_estimators=n, max_depth=d, max_features=f)
                    rf.fit(trnX, trnY)
                    prdY = rf.predict(tstX)
                    yvalues.append(precision_score(tstY, prdY))
                    if yvalues[-1] > last_best:
                        best = (d, f, n)
                        last_best = yvalues[-1]
                        best_model = rf

                values[f] = yvalues
            multiple_line_chart(n_estimators, values, ax=axs[0, k], title=f'Random Forests with max_depth={d}',
                                xlabel='nr estimators', ylabel='precision', percentage=True)
        savefig(f'images/random_forests/rf_study.png')
        #show()
        print('Best results with depth=%d, %1.2f features and %d estimators, with precision=%1.2f' % (
        best[0], best[1], best[2], last_best))

        best = (25, 0.70, 100)
        best_model = RandomForestClassifier(n_estimators=best[2], max_depth=best[0], max_features=best[1])
        best_model.fit(trnX, trnY)

        prd_trn = best_model.predict(trnX)
        prd_tst = best_model.predict(tstX)
        plot_evaluation_results(["Safe", "Danger"], trnY, prd_trn, tstY, prd_tst, extra="")
        savefig(f'images/random_forests/rf_best.png')
        #show()


        from numpy import std, argsort

        variables = trnX.columns
        importances = best_model.feature_importances_
        stdevs = std([tree.feature_importances_ for tree in best_model.estimators_], axis=0)
        indices = argsort(importances)[::-1]
        elems = []
        for f in range(len(variables)):
            elems += [variables[indices[f]]]
            print(f'{f + 1}. feature {elems[f]} ({importances[indices[f]]})')

        figure()
        horizontal_bar_chart(elems, importances[indices], stdevs[indices], title='Random Forest Features importance',
                             xlabel='importance', ylabel='variables')
        savefig(f'images/random_forests/rf_ranking.png')


        # Overfitting
        def plot_overfitting_study(xvalues, prd_trn, prd_tst, name, xlabel, ylabel, extra=""):
            evals = {'Train': prd_trn, 'Test': prd_tst}
            plt.figure()
            multiple_line_chart(xvalues, evals, ax=None, title=f'Overfitting {name}', xlabel=xlabel, ylabel=ylabel,
                                percentage=True)
            plt.savefig(f'images/random_forests/overfitting_{extra}.png')

        print("overfitting")
        f = best[1]
        max_depth = best[0]
        y_tst_values = []
        y_trn_values = []
        for n in n_estimators:
            print("n: ", n)
            rf = RandomForestClassifier(n_estimators=n, max_depth=max_depth, max_features=f)
            rf.fit(trnX, trnY)
            prd_tst_Y = rf.predict(tstX)
            prd_trn_Y = rf.predict(trnX)
            y_tst_values.append(precision_score(tstY, prd_tst_Y))
            y_trn_values.append(precision_score(trnY, prd_trn_Y))
        plot_overfitting_study(n_estimators, y_trn_values, y_tst_values, name=f'RF_depth={max_depth}_vars={f}',
                               xlabel='nr_estimators', ylabel="precision")

        f = open(f"images/random_forests/bestModel.txt", "w")
        f.write(
            'Best results achieved with %s criteria, depth=%d and min_impurity_decrease=%1.2f ==> precision=%1.2f' % (
                best[0], best[1], best[2], last_best))
        f.close()


