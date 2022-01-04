import pandas as pd
from matplotlib.pyplot import figure, savefig, show
from pandas import read_csv, unique, DataFrame, concat
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.preprocessing import MinMaxScaler

from NaiveBayes_Set2.balanceData_set2 import BalanceData
from ds_charts import plot_evaluation_results, bar_chart, get_variable_types

file_tag = 'NoBalancing'
filename = '../MissingValuesImputation_Set2/data/dataset2_mv_most_frequent.csv'

df = read_csv(f'{filename}')
print("shape: ", df.shape)


symbolic_vars = get_variable_types(df)['Symbolic']
for symbolic_var in symbolic_vars:
    df[symbolic_var] = pd.factorize(df[symbolic_var])[0]

binary_vars = get_variable_types(df)['Binary']
for binary_var in binary_vars:
    df[binary_var] = pd.factorize(df[binary_var])[0]



variable_types = get_variable_types(df)
numeric_vars = variable_types['Numeric']
symbolic_vars = variable_types['Symbolic']
boolean_vars = variable_types['Binary']

df_nr = df[numeric_vars]
df_sb = df[symbolic_vars]
df_bool = df[boolean_vars]


transf = MinMaxScaler(feature_range=(0, 1), copy=True).fit(df_nr)
tmp = DataFrame(transf.transform(df_nr), index=df.index, columns=numeric_vars)
df = concat([tmp, df_sb, df_bool], axis=1)


trnX, tstX, trnY, tstY = train_test_split(df, df['ALARM'], test_size=0.3, random_state=1)
balanceData = BalanceData(dataframe=pd.concat([trnX], axis=1))
#result_smote = balanceData.pos_sample()
#trnX = result_smote.copy()
#trnY = result_smote['ALARM']



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
show()

figure()
bar_chart(xvalues, yvalues, title='Comparison of Naive Bayes Models', ylabel='accuracy', percentage=True)
savefig(f'images/{file_tag}_nb_study.png')
show()