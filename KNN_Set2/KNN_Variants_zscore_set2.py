import pandas as pd
from matplotlib.pyplot import figure, savefig, show
from pandas import read_csv, unique
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from ds_charts import plot_evaluation_results, multiple_line_chart, get_variable_types

file_tag = 'air_quality_tabular_scaled_zscore'
filename = '../Scaling_Set2/data/air_quality_tabular_scaled_zscore.csv'
target = 'ALARM'



df = read_csv(f'{filename}')
df = df.sample(40000)

df = df.dropna()
print("shape : ", df.shape)

symbolic_vars = get_variable_types(df)['Symbolic']
for symbolic_var in symbolic_vars:
    df[symbolic_var] = pd.factorize(df[symbolic_var])[0]

binary_vars = get_variable_types(df)['Binary']
for binary_var in binary_vars:
    df[binary_var] = pd.factorize(df[binary_var])[0]

trnX, tstX, trnY, tstY = train_test_split(df, df['ALARM'], test_size=0.3, random_state=1)
labels = unique(trnY)
labels.sort()

nvalues = [7, 9, 15]
dist = ['manhattan', 'euclidean', 'chebyshev', 'minkowski', 'wminkowski']
values = {}
best = (0, '')
last_best = 0
for d in dist:
    print("d -> ", d)
    yvalues = []
    for n in nvalues:
        print("n -> ", n)
        if d == 'wminkowski':
            knn = KNeighborsClassifier(n_neighbors=n, metric=d, p=2, metric_params={'w': 3})
        else:
            knn = KNeighborsClassifier(n_neighbors=n, metric=d, p=2)
        """        
        elif d == 'seuclidean' or d == 'mahalanobis':
            knn = KNeighborsClassifier(n_neighbors=n, metric=d, p=2, metric_params={'V': 4})
        """
        knn.fit(trnX, trnY)
        prdY = knn.predict(tstX)
        yvalues.append(accuracy_score(tstY, prdY))
        if yvalues[-1] > last_best:
            best = (n, d)
            last_best = yvalues[-1]
    values[d] = yvalues

figure()
multiple_line_chart(nvalues, values, title='KNN variants', xlabel='n', ylabel='accuracy', percentage=True)
savefig(f'images/{file_tag}_knn_study.png')
show()
print('Best results with %d neighbors and %s'%(best[0], best[1]))

clf = knn = KNeighborsClassifier(n_neighbors=best[0], metric=best[1])
clf.fit(trnX, trnY)
prd_trn = clf.predict(trnX)
prd_tst = clf.predict(tstX)
plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst)
savefig(f'images/{file_tag}_knn_best.png')
show()