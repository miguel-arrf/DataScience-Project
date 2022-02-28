from matplotlib import pyplot as plt
from matplotlib.pyplot import show, figure, savefig, subplots
from numpy import argsort
from pandas import unique
from sklearn.metrics import recall_score
from sklearn.tree import DecisionTreeClassifier

from ds_charts import multiple_line_chart, horizontal_bar_chart, plot_evaluation_results


class DecisionTrees:
    def __init__(self, trnX, trnY, tstY, tstX, imagePath):
        labels = unique(trnY)
        labels.sort()

        min_impurity_decrease = [0.01, 0.005, 0.0025, 0.001, 0.0005]
        max_depths = [2, 5, 10, 15, 20]
        criteria = ['entropy', 'gini']
        best = ('', 0, 0.0)
        last_best = 0
        best_model = None

        figure()
        fig, axs = subplots(1, 2, figsize=(10, 4), squeeze=False)
        for k in range(len(criteria)):
            f = criteria[k]
            print("k: ", f)
            values = {}
            for d in max_depths:
                print("\td: ", d)
                yvalues = []
                for imp in min_impurity_decrease:
                    print("\t\timp: ", imp)
                    tree = DecisionTreeClassifier(max_depth=d, criterion=f, min_impurity_decrease=imp)
                    tree.fit(trnX, trnY)
                    prdY = tree.predict(tstX)
                    yvalues.append(recall_score(tstY, prdY, pos_label="Killed"))
                    if yvalues[-1] > last_best:
                        best = (f, d, imp)
                        last_best = yvalues[-1]
                        best_model = tree

                values[d] = yvalues
            multiple_line_chart(min_impurity_decrease, values, ax=axs[0, k], title=f'Decision Trees with {f} criteria',
                                xlabel='min_impurity_decrease', ylabel='recall', percentage=True)
        savefig(f'{imagePath}/decision_trees/dt_study.png', dpi=300)
        show()
        print('Best results achieved with %s criteria, depth=%d and min_impurity_decrease=%1.4f ==> recall=%1.2f' % (
        best[0], best[1], best[2], last_best))

        from sklearn import tree

        labels = [str(value) for value in labels]
        figure(figsize=(72, 72), dpi=80)
        tree.plot_tree(best_model, feature_names=trnX.columns, class_names=labels)
        savefig(f'{imagePath}/decision_trees/dt_best_tree.png', dpi=300)

        prd_trn = best_model.predict(trnX)
        prd_tst = best_model.predict(tstX)
        figure()
        plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst, extra="Decision Tree")
        savefig(f'{imagePath}/decision_trees/dt_best.png', dpi=300)
        show()


        variables = trnX.columns
        importances = best_model.feature_importances_
        indices = argsort(importances)[::-1]
        elems = []
        imp_values = []
        for f in range(len(variables)):
            elems += [variables[indices[f]]]
            imp_values += [importances[indices[f]]]
            print(f'{f + 1}. feature {elems[f]} ({importances[indices[f]]})')

        figure(figsize=(1*4, 4))
        horizontal_bar_chart(elems[0: 5], imp_values[0:5], error=None, title='Decision Tree Features importance',
                             xlabel='importance', ylabel='variables')
        plt.tight_layout()
        savefig(f'{imagePath}/decision_trees/dt_ranking.png', dpi=300)
        show()

        def plot_overfitting_study(xvalues, prd_trn, prd_tst, name, xlabel, ylabel, extra=""):
            evals = {'Train': prd_trn, 'Test': prd_tst}
            plt.figure()
            multiple_line_chart(xvalues, evals, ax=None, title=f'Overfitting {name}', xlabel=xlabel, ylabel=ylabel,
                                percentage=True)
            plt.tight_layout()
            plt.savefig(f'{imagePath}/decision_trees/overfitting_{extra}.png', dpi=300)

        imp = best[2]
        f = best[0]
        y_tst_values = []
        y_trn_values = []
        for d in max_depths:
            tree = DecisionTreeClassifier(max_depth=d, criterion=f, min_impurity_decrease=imp)
            tree.fit(trnX, trnY)
            prdY = tree.predict(tstX)
            prd_tst_Y = tree.predict(tstX)
            prd_trn_Y = tree.predict(trnX)
            y_tst_values.append(recall_score(tstY, prd_tst_Y, pos_label="Killed"))
            y_trn_values.append(recall_score(trnY, prd_trn_Y, pos_label="Killed"))
        plot_overfitting_study(max_depths, y_trn_values, y_tst_values, name=f'DT=imp{imp}_{f}', xlabel='max_depth',
                               ylabel="recall")

        f = open(f"{imagePath}/decision_trees/bestModel.txt", "w")
        f.write('Best results achieved with %s criteria, depth=%d and min_impurity_decrease=%1.4f ==> recall=%1.2f' % (
        best[0], best[1], best[2], last_best))
        f.close()


