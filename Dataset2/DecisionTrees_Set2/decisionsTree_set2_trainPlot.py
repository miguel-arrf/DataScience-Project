from subprocess import call

import pandas as pd
from matplotlib.pyplot import figure, subplots, savefig, show
from pandas import read_csv, unique
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from ds_charts import get_variable_types, multiple_line_chart, plot_evaluation_results

target = 'ALARM'
df = read_csv(f'../../data/air_quality_tabular.csv')
df = df.dropna()
symbolic_vars = get_variable_types(df)['Symbolic']
for symbolic_var in symbolic_vars:
    df[symbolic_var] = pd.factorize(df[symbolic_var])[0]

binary_vars = get_variable_types(df)['Binary']
for binary_var in binary_vars:
    df[binary_var] = pd.factorize(df[binary_var])[0]
y = df['ALARM']
df = df.drop('ALARM', 1)

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
fig, axs = subplots(3, 2, figsize=(16, 8), dpi=150, squeeze=False)
fig.tight_layout(pad=3.0)

row = 0
for score in [(accuracy_score, 'accuracy'), (precision_score, 'precision'), (recall_score, 'recall')]:

    for k in range(len(criteria)):
        f = criteria[k]
        values = {}
        for d in max_depths:
            yvalues = []
            for imp in min_impurity_decrease:
                tree = DecisionTreeClassifier(max_depth=d, criterion=f, min_impurity_decrease=imp)
                tree.fit(trnX, trnY)
                prdY = tree.predict(trnX)
                print("{} : {}".format(score[1], score[0](trnY, prdY)))
                yvalues.append(score[0](trnY, prdY))

                if yvalues[-1] > last_best:
                    best = (f, d, imp, prdY) 
                    last_best = yvalues[-1]
                    best_model = tree

            values[d] = yvalues
        multiple_line_chart(min_impurity_decrease, values, ax=axs[row, k],
                            title=f'Decision Trees with {f} criteria - {score[1]} - @train',
                            xlabel='min_impurity_decrease', ylabel="{}".format(score[1]), percentage=True)
    row += 1
savefig(f"images/train_decisionTree.png")
# show()
print('Best results achieved with %s criteria, depth=%d and min_impurity_decrease=%1.2f ==> accuracy=%1.2f' % (
    best[0], best[1], best[2], last_best))

print(confusion_matrix(trnY, best[3]))

predictions = best_model.predict(trnX)
cm = confusion_matrix(trnY, predictions, labels=best_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=best_model.classes_)
disp.plot()
savefig(f'images/train_confusionMatrix_bestModel.png')

show()

