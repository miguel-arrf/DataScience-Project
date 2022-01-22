from matplotlib import pyplot as plt
from matplotlib.pyplot import figure, subplots
from pandas import read_csv, unique, factorize
from sklearn.metrics import confusion_matrix, classification_report, \
    precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from ds_charts import get_variable_types, multiple_line_chart

target = 'ALARM'
df = read_csv(f'../MissingValuesImputation_Set2/data/dataset2_mv_most_frequent.csv')


filename = '../MissingValuesImputation_Set2/data/dataset2_mv_most_frequent.csv'

df = read_csv(f'{filename}')
df = df.sample(frac=1)

df = df.dropna()

symbolic_vars = get_variable_types(df)['Symbolic']
for symbolic_var in symbolic_vars:
    df[symbolic_var] = factorize(df[symbolic_var])[0]

binary_vars = get_variable_types(df)['Binary']
for binary_var in binary_vars:
    df[binary_var] = factorize(df[binary_var])[0]

y = df[target]
df = df.drop([target], axis=1)
trnX, tstX, trnY, tstY = train_test_split(df, y, test_size=0.3, random_state=1,
                                          stratify=y)
labels = unique(trnY)
labels.sort()


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
tree = DecisionTreeClassifier(max_depth=5, criterion='entropy', min_impurity_decrease=0.0025)
tree.fit(trnX, trnY)
best_model = tree
prdY = tree.predict(tstX)
best = ('entropy', 5, 0.0025, prdY)
print('Best results achieved with %s criteria, depth=%d and min_impurity_decrease=%1.2f ==> accuracy=%1.2f' % (
    best[0], best[1], best[2], last_best))

print(confusion_matrix(tstY, best[3]))
print(classification_report(tstY, best[3]))

figure(figsize=(14, 18))
labels = unique(trnY)
labels.sort()
labels = [str(value) for value in labels]

prd_trn = best_model.predict(trnX)
prd_tst = best_model.predict(tstX)

# Feature importance:

def plot_overfitting_study(xvalues, prd_trn, prd_tst, name, xlabel, ylabel, extra=""):
    evals = {'Train': prd_trn, 'Test': prd_tst}
    plt.figure()
    multiple_line_chart(xvalues, evals, ax = None, title=f'Overfitting {name}', xlabel=xlabel, ylabel=ylabel, percentage=True)
    plt.savefig(f'images/overfitting_{extra}.png')

imp = 0.0025
f = 'entropy'
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
plot_overfitting_study(max_depths, y_trn_values, y_tst_values, name=f'DT=imp{imp}_{f}', xlabel='max_depth', ylabel=str("precision"), extra="precision")



imp = 0.0025
f = 'entropy'
y_tst_values = []
y_trn_values = []
for d in max_depths:
    tree = DecisionTreeClassifier(max_depth=d, criterion=f, min_impurity_decrease=imp)
    tree.fit(trnX, trnY)
    prdY = tree.predict(tstX)
    prd_tst_Y = tree.predict(tstX)
    prd_trn_Y = tree.predict(trnX)
    y_tst_values.append(recall_score(tstY, prd_tst_Y))
    y_trn_values.append(recall_score(trnY, prd_trn_Y))
plot_overfitting_study(max_depths, y_trn_values, y_tst_values, name=f'DT=imp{imp}_{f}', xlabel='max_depth', ylabel=str("recall"), extra="recall")