from subprocess import call

import pandas as pd
from matplotlib.pyplot import figure, subplots, savefig, show
from pandas import read_csv, unique
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, \
    precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from ds_charts import get_variable_types, multiple_line_chart, plot_evaluation_results

target = 'PERSON_INJURY'
df = read_csv(f'../data/for_tree_labs.csv')
y = df['PERSON_INJURY']
df = df.drop('PERSON_INJURY', 1)
df["PERSON_SEX"].replace(('F', 'M'), (1, 0), inplace=True)

#y = y.replace({0: 1, 1: 0})
trnX, tstX, trnY, tstY = train_test_split(df, y, test_size=0.3, random_state=1,
                                          stratify=y)

print("columns: ", trnX.columns)
print("testX: ", tstX)
print("--")
print("testY: ", tstY)

min_impurity_decrease = [0.01, 0.005, 0.0025, 0.001, 0.0005]
max_depths = [2, 5, 10, 15, 20, 25]
criteria = ['entropy', 'gini']
best = ('', 0, 0.0, None)
last_best = 0
best_model = None




figure()
fig, axs = subplots(2, 2, figsize=(16, 8), dpi=150, squeeze=False)
fig.tight_layout(pad=3.0)

row = 0
#(accuracy_score, 'accuracy')
for score in [ (precision_score, 'precision'), (recall_score, 'recall')]:

    for k in range(len(criteria)):
        f = criteria[k]
        values = {}
        for d in max_depths:
            yvalues = []
            for imp in min_impurity_decrease:
                tree = DecisionTreeClassifier(max_depth=d, criterion=f, min_impurity_decrease=imp)
                tree.fit(trnX, trnY)
                prdY = tree.predict(tstX)

                if score[1] != "accuracy":
                    yvalues.append(score[0](tstY, prdY, pos_label="Killed"))
                else:
                    yvalues.append(score[0](tstY, prdY))

                print("{} : {}".format(score[1], score[0](tstY, prdY, pos_label="Killed")))

                if yvalues[-1] > last_best:
                    best = (f, d, imp, prdY)
                    last_best = yvalues[-1]
                    best_model = tree

            values[d] = yvalues
        multiple_line_chart(min_impurity_decrease, values, ax=axs[row, k],
                            title=f'Decision Trees with {f} criteria - {score[1]} - @test',
                            xlabel='min_impurity_decrease', ylabel="{}".format(score[1]), percentage=True)
    row += 1
savefig(f'images/test_decisionTree.png')
# show()
print('Best results achieved with %s criteria, depth=%d and min_impurity_decrease=%1.2f ==> accuracy=%1.2f' % (
    best[0], best[1], best[2], last_best))

print(confusion_matrix(tstY, best[3]))
print(classification_report(tstY, best[3]))

from sklearn import tree

figure(figsize=(14, 18))
labels = unique(trnY)
labels.sort()
labels = [str(value) for value in labels]
print("labels: ", labels)
print("columns: ", df.columns)
tree.plot_tree(best_model, feature_names=trnX.columns, class_names=labels)
savefig(f'images/test_treeView_bestTree.png')

predictions = best_model.predict(tstX)
cm = confusion_matrix(tstY, predictions, labels=best_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
disp.plot()
show()

prd_trn = best_model.predict(trnX)
prd_tst = best_model.predict(tstX)
plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst)
savefig(f'images/test_confusionMatrix_bestTree.png')
show()

# Feature importance:
from numpy import argsort
from ds_charts import horizontal_bar_chart

variables = trnX.columns
importances = best_model.feature_importances_
indices = argsort(importances)[::-1]
elems = []
imp_values = []
for f in range(len(variables)):
    elems += [variables[indices[f]]]
    imp_values += [importances[indices[f]]]
    print(f'{f + 1}. feature {elems[f]} ({importances[indices[f]]})')

figure(figsize=(14, 6), dpi=200)
horizontal_bar_chart(elems, imp_values, error=None, title='Decision Tree Features importance', xlabel='importance',
                     ylabel='variables')
savefig(f'images/test_featureImportance_bestTree.png')