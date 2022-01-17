import pandas as pd
from matplotlib.pyplot import figure, savefig
from pandas import read_csv, unique
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

from ds_charts import plot_evaluation_results, bar_chart, get_variable_types

file_tag = 'air_quality_tabular'
filename = './data/redundant_removed.csv'

df = read_csv(f'{filename}')
df = df.dropna()
print("shape: ", df.shape)


symbolic_vars = get_variable_types(df)['Symbolic']
for symbolic_var in symbolic_vars:
    df[symbolic_var] = pd.factorize(df[symbolic_var])[0]

binary_vars = get_variable_types(df)['Binary']
for binary_var in binary_vars:
    df[binary_var] = pd.factorize(df[binary_var])[0]

trnX, tstX, trnY, tstY = train_test_split(df, df['ALARM'], test_size=0.3, random_state=1, stratify=df['ALARM'])
labels = unique(trnY)
labels.sort()


estimators = {'GaussianNB': GaussianNB(),
              'MultinomialNB1': MultinomialNB(fit_prior=False),
              'MultinomialNB2': MultinomialNB(),
              'BernoulliNB': BernoulliNB()
              }

labels = unique(trnY)
labels.sort()

xvalues = []
yvalues = []
acc = 0.0
best = None
for clf in estimators:
    xvalues.append(clf)
    estimators[clf].fit(trnX, trnY)
    prdY = estimators[clf].predict(tstX)
    prd_trn = estimators[clf].predict(trnX)
    prd_tst = estimators[clf].predict(tstX)
    plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst)
    savefig(f'images/{file_tag}_{estimators[clf]}.png')
    yvalues.append(accuracy_score(tstY, prdY))
    if accuracy_score(tstY,prdY) > acc:
        best = clf


estimators[best].fit(trnX, trnY)
prd_trn = estimators[best].predict(trnX)
prd_tst = estimators[best].predict(tstX)
plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst)
savefig(f'images/{file_tag}_nb_best.png')


figure()
bar_chart(xvalues, yvalues, title='Comparison of Naive Bayes Models', ylabel='accuracy', percentage=True)
savefig(f'images/{file_tag}_nb_study.png')
