from matplotlib import pyplot as plt
from matplotlib.pyplot import show, figure, savefig, subplots
from numpy import argsort, std
from pandas import unique
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score

from ds_charts import multiple_line_chart, HEIGHT, plot_evaluation_results, horizontal_bar_chart


class GradientBoosting:
    def __init__(self, trnX, trnY, tstY, tstX):
        labels = unique(trnY)
        labels.sort()

        n_estimators = [5, 10, 25, 50, 75, 100, 200, 300, 400]
        max_depths = [5, 10, 25]
        learning_rate = [.1, .5, .9]
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
            for lr in learning_rate:
                print("\tlr: ", lr)
                yvalues = []
                for n in n_estimators:
                    print("\t\tn: ", n)
                    gb = GradientBoostingClassifier(n_estimators=n, max_depth=d, learning_rate=lr)
                    gb.fit(trnX, trnY)
                    prdY = gb.predict(tstX)
                    yvalues.append(precision_score(tstY, prdY, pos_label="Killed"))
                    if yvalues[-1] > last_best:
                        best = (d, lr, n)
                        last_best = yvalues[-1]
                        best_model = gb
                values[lr] = yvalues
            multiple_line_chart(n_estimators, values, ax=axs[0, k], title=f'Gradient Boorsting with max_depth={d}',
                                xlabel='nr estimators', ylabel='precision', percentage=True)
        savefig(f'images/gradient_boosting/gb_study.png')
        show()


        #best = (10, 0.9, 300)
        #best_model = GradientBoostingClassifier(n_estimators=300, max_depth=10, learning_rate=0.9)
        #best_model.fit(trnX, trnY)

        print('Best results with depth=%d, learning rate=%1.2f and %d estimators, with precision=%1.2f' % (
            best[0], best[1], best[2], last_best))

        # Best model:
        prd_trn = best_model.predict(trnX)
        prd_tst = best_model.predict(tstX)
        plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst, extra="")
        savefig(f'images/gradient_boosting/gb_best.png')
        show()


        # Feature Importance:
        variables = trnX.columns
        importances = best_model.feature_importances_
        indices = argsort(importances)[::-1]
        stdevs = std([tree[0].feature_importances_ for tree in best_model.estimators_], axis=0)
        elems = []
        for f in range(len(variables)):
            elems += [variables[indices[f]]]
            print(f'{f + 1}. feature {elems[f]} ({importances[indices[f]]})')

        figure()
        horizontal_bar_chart(elems, importances[indices], stdevs[indices],
                             title='Gradient Boosting Features importance', xlabel='importance', ylabel='variables')
        savefig(f'images/gradient_boosting/gb_ranking.png')

        # Overfitting
        def plot_overfitting_study(xvalues, prd_trn, prd_tst, name, xlabel, ylabel, extra=""):
            evals = {'Train': prd_trn, 'Test': prd_tst}
            plt.figure()
            multiple_line_chart(xvalues, evals, ax=None, title=f'Overfitting {name}', xlabel=xlabel, ylabel=ylabel,
                                percentage=True)
            plt.savefig(f'images/gradient_boosting/overfitting_{extra}.png')

        lr = best[1]
        max_depth = best[0]
        y_tst_values = []
        y_trn_values = []
        for n in n_estimators:
            gb = GradientBoostingClassifier(n_estimators=n, max_depth=max_depth, learning_rate=lr)
            gb.fit(trnX, trnY)
            prd_tst_Y = gb.predict(tstX)
            prd_trn_Y = gb.predict(trnX)
            y_tst_values.append(precision_score(tstY, prd_tst_Y, pos_label="Killed"))
            y_trn_values.append(precision_score(trnY, prd_trn_Y, pos_label="Killed"))
        plot_overfitting_study(n_estimators, y_trn_values, y_tst_values, name=f'GB_depth={max_depth}_lr={lr}',
                               xlabel='nr_estimators', ylabel="precision")

        f = open(f"images/gradient_boosting/bestModel.txt", "w")
        f.write('Best results with depth=%d, learning rate=%1.2f and %d estimators, with precision=%1.2f' % (
            best[0], best[1], best[2], last_best))
        f.close()

        f = open(f"images/gradient_boosting/bestModel.txt", "w")
        f.write('Best results with depth=%d, learning rate=%1.2f and %d estimators, with precision=%1.2f' % (
            best[0], best[1], best[2], last_best))
        f.close()
