from matplotlib import pyplot as plt
from numpy import ndarray, std, argsort
from pandas import DataFrame, read_csv, unique
from matplotlib.pyplot import figure, subplots, savefig, show
from sklearn.ensemble import RandomForestClassifier
from ds_charts import plot_evaluation_results, multiple_line_chart, horizontal_bar_chart, HEIGHT
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

target = 'ALARM'
#df = read_csv(f'./data/redundant_removed.csv')
df = read_csv(f'../../data/air_quality_tabular.csv')
df = df.dropna()
y = df['ALARM']
df = df.drop('ALARM', 1)

# y = y.replace({0: 1, 1: 0})
trnX, tstX, trnY, tstY = train_test_split(df, y, test_size=0.3, random_state=1,
                                          stratify=y)
labels = unique(trnY)
labels.sort()

n_estimators = [5, 25, 150]
max_depths = [5, 25]
max_features = [.5, .9, 1]
best = ('', 0, 0)
last_best = 0
best_model = None


def plot_overfitting_study(xvalues, prd_trn, prd_tst, name, xlabel, ylabel):
    evals = {'Train': prd_trn, 'Test': prd_tst}
    plt.figure()
    multiple_line_chart(xvalues, evals, ax=None, title=f'Overfitting {name}', xlabel=xlabel, ylabel=ylabel,
                        percentage=True)
    plt.savefig(f'images/overfitting_{name}.png')


cols = len(max_depths)
figure()
fig, axs = subplots(1, cols, figsize=(cols * HEIGHT, HEIGHT), squeeze=False)
for k in range(len(max_depths)):
    print("k: ", k)
    d = max_depths[k]
    values = {}
    for f in max_features:
        print("f: ", f)
        yvalues = []

        y_tst_values = []
        y_trn_values = []
        eval_metric = accuracy_score
        max_depth = d

        for n in n_estimators:
            print("n: ", n)
            rf = RandomForestClassifier(n_estimators=n, max_depth=d, max_features=f)
            rf.fit(trnX, trnY)
            prdY = rf.predict(tstX)
            yvalues.append(accuracy_score(tstY, prdY))
            if yvalues[-1] > last_best:
                best = (d, f, n)
                last_best = yvalues[-1]
                best_model = rf
            prd_tst_Y = rf.predict(tstX)
            prd_trn_Y = rf.predict(trnX)
            y_tst_values.append(eval_metric(tstY, prd_tst_Y))
            y_trn_values.append(eval_metric(trnY, prd_trn_Y))
        plot_overfitting_study(n_estimators, y_trn_values, y_tst_values, name=f'RF_depth={max_depth}_vars={f}',
                               xlabel='nr_estimators', ylabel=str(eval_metric))

        values[f] = yvalues
    # multiple_line_chart(n_estimators, values, ax=axs[0, k], title=f'Random Forests with max_depth={d}',
    #                       xlabel='nr estimators', ylabel='accuracy', percentage=True)
savefig(f'images/rf_study.png')
# show()
print('Best results with depth=%d, %1.2f features and %d estimators, with accuracy=%1.2f' % (
best[0], best[1], best[2], last_best))

prd_trn = best_model.predict(trnX)
prd_tst = best_model.predict(tstX)
plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst)
savefig(f'images/rf_best.png')
# show()


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
savefig(f'images/rf_ranking.png')


f = 0.7
max_depth = 10
eval_metric = accuracy_score
y_tst_values = []
y_trn_values = []
for n in n_estimators:
    rf = RandomForestClassifier(n_estimators=n, max_depth=best[0], max_features=best[1])
    rf.fit(trnX, trnY)
    prd_tst_Y = rf.predict(tstX)
    prd_trn_Y = rf.predict(trnX)
    y_tst_values.append(eval_metric(tstY, prd_tst_Y))
    y_trn_values.append(eval_metric(trnY, prd_trn_Y))
plot_overfitting_study(n_estimators, y_trn_values, y_tst_values, name=f'RF_depth={max_depth}_vars={f}', xlabel='nr_estimators', ylabel=str(eval_metric))