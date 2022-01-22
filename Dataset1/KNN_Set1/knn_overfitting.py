from numpy import ndarray
from pandas import DataFrame, read_csv, unique
from matplotlib.pyplot import figure, savefig, show
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from ds_charts import plot_evaluation_results, multiple_line_chart, plot_overfitting_study
from sklearn.metrics import recall_score, precision_score, accuracy_score

target = 'PERSON_INJURY'
df = read_csv(f'../teste_to_use.csv')

y = df['PERSON_INJURY']
df = df.drop('PERSON_INJURY', 1)
df["PERSON_SEX"].replace(('F', 'M'), (1, 0), inplace=True)

trnX, tstX, trnY, tstY = train_test_split(df, y, test_size=0.3, random_state=1, stratify=y)

labels = unique(trnY)
labels.sort()

eval_metric = recall_score
nvalues = [5, 9, 11, 12, 19]

from matplotlib.pyplot import figure, savefig


def plot_overfitting_study(xvalues, prd_trn, prd_tst, name, xlabel, ylabel, extra=""):
    evals = {'Train': prd_trn, 'Test': prd_tst}
    figure()
    multiple_line_chart(xvalues, evals, ax=None, title=f'Overfitting {name}', xlabel=xlabel, ylabel=ylabel,
                        percentage=True)
    savefig(f'images/overfitting_{name}_{extra}.png')


d = 'euclidean'
eval_metric = recall_score
y_tst_values = []
y_trn_values = []
for n in nvalues:
    knn = KNeighborsClassifier(n_neighbors=6, metric='euclidean')
    knn.fit(trnX, trnY)
    prd_tst_Y = knn.predict(tstX)
    prd_trn_Y = knn.predict(trnX)
    y_tst_values.append(accuracy_score(tstY, prd_tst_Y))
    y_trn_values.append(accuracy_score(trnY, prd_trn_Y))
plot_overfitting_study(nvalues, y_trn_values, y_tst_values, name=f'KNN_K={d}', xlabel='K', ylabel="recall",
                       extra="recall")
