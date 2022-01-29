from matplotlib import pyplot as plt
from matplotlib.pyplot import show, figure, savefig, subplots
from numpy import argsort
from pandas import unique
from sklearn.metrics import precision_score
from sklearn.tree import DecisionTreeClassifier

from ds_charts import multiple_line_chart, horizontal_bar_chart, plot_evaluation_results


class DecisionTrees:
    def  __init__(self, trnX, trnY, tstY, tstX):
        labels = unique(trnY)
        labels.sort()

        min_impurity_decrease = [0.01, 0.005, 0.0025, 0.001, 0.0005]
        max_depths = [2, 5, 10, 15, 20, 25]
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
                    yvalues.append(precision_score(tstY, prdY))
                    if yvalues[-1] > last_best:
                        best = (f, d, imp)
                        last_best = yvalues[-1]
                        best_model = tree

                values[d] = yvalues
            multiple_line_chart(min_impurity_decrease, values, ax=axs[0, k], title=f'Decision Trees with {f} criteria',
                                xlabel='min_impurity_decrease', ylabel='precision', percentage=True)
        savefig(f'images/decision_trees/dt_study.png')
        #show()
        print('Best results achieved with %s criteria, depth=%d and min_impurity_decrease=%1.4f ==> precision=%1.2f' % (
        best[0], best[1], best[2], last_best))

        from sklearn import tree

        labels = [str(value) for value in labels]
        figure(figsize=(72, 72), dpi=80)
        tree.plot_tree(best_model, feature_names=trnX.columns, class_names=labels)
        savefig(f'images/decision_trees/dt_best_tree.png')

        prd_trn = best_model.predict(trnX)
        prd_tst = best_model.predict(tstX)
        figure()
        plot_evaluation_results(["Safe", "Danger"], trnY, prd_trn, tstY, prd_tst, extra="")
        savefig(f'images/decision_trees/dt_best.png')
        #show()


        variables = trnX.columns
        importances = best_model.feature_importances_
        indices = argsort(importances)[::-1]
        elems = []
        imp_values = []
        for f in range(len(variables)):
            elems += [variables[indices[f]]]
            imp_values += [importances[indices[f]]]
            print(f'{f + 1}. feature {elems[f]} ({importances[indices[f]]})')

        figure(figsize=(8, 6), dpi=140)
        horizontal_bar_chart(elems[0: 5], imp_values[0:5], error=None, title='Decision Tree Features importance',
                             xlabel='importance', ylabel='variables')
        savefig(f'images/decision_trees/dt_ranking.png')

        def plot_overfitting_study(xvalues, prd_trn, prd_tst, name, xlabel, ylabel, extra=""):
            evals = {'Train': prd_trn, 'Test': prd_tst}
            plt.figure()
            multiple_line_chart(xvalues, evals, ax=None, title=f'Overfitting {name}', xlabel=xlabel, ylabel=ylabel,
                                percentage=True)
            plt.savefig(f'images/decision_trees/overfitting_{extra}.png')

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
            y_tst_values.append(precision_score(tstY, prd_tst_Y))
            y_trn_values.append(precision_score(trnY, prd_trn_Y))
        plot_overfitting_study(max_depths, y_trn_values, y_tst_values, name=f'DT=imp{imp}_{f}', xlabel='max_depth',
                               ylabel="precision")

        f = open(f"images/decision_trees/bestModel.txt", "w")
        f.write('Best results achieved with %s criteria, depth=%d and min_impurity_decrease=%1.4f ==> precision=%1.2f' % (
        best[0], best[1], best[2], last_best))
        f.close()


