import pandas as pd
from matplotlib.pyplot import figure, savefig, show
from pandas import read_csv, unique, DataFrame, concat
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

from ds_charts import plot_evaluation_results, bar_chart, get_variable_types

file_tag = 'NoBalancing'
filename = '../MissingValuesImputation_Set1/data/dataset1_mv_most_frequent.csv'
filename = '../teste_to_use.csv'

df = read_csv(f'{filename}')
print("shape: ", df.shape)

var = 'PERSON_INJURY'

killed = df.loc[df[var] == 'Killed']
injured = df.loc[df[var] == 'Injured']
df = pd.concat([killed, injured], axis=0)
print("new shape: ", df.shape)

'''
symbolic_vars = get_variable_types(df)['Symbolic']
for symbolic_var in symbolic_vars:
    df[symbolic_var] = pd.factorize(df[symbolic_var])[0]

binary_vars = get_variable_types(df)['Binary']
for binary_var in binary_vars:
    df[binary_var] = pd.factorize(df[binary_var])[0]
'''
df["PERSON_SEX"] = pd.factorize(df["PERSON_SEX"])[0]

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

y = df['PERSON_INJURY']
X = df.drop(["PERSON_INJURY"], axis=1)
trnX, tstX, trnY, tstY = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
print(trnX.columns)


labels = unique(trnY)
labels.sort()



estimators = {'NaiveBayes': KNeighborsClassifier(n_neighbors=5, metric='manhattan')
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
    plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst, extra="teste1")
    savefig(f'images/{file_tag}_{estimators[clf]}.png')
    yvalues.append(accuracy_score(tstY, prdY))
    if accuracy_score(tstY,prdY) > acc:
        best = clf


estimators[best].fit(trnX, trnY)
prd_trn = estimators[best].predict(trnX)
prd_tst = estimators[best].predict(tstX)
plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst, extra="Teste2")
savefig(f'images/{file_tag}_nb_best.png')
show()

figure()
bar_chart(xvalues, yvalues, title='Comparison of Naive Bayes Models', ylabel='accuracy', percentage=True)
savefig(f'images/{file_tag}_nb_study.png')
show()