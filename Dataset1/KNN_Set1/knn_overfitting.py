import pandas as pd
from matplotlib.pyplot import figure, savefig, show, subplots
from pandas import DataFrame, read_csv, unique, concat
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

from Dataset1.NaiveBayes_Set1.balanceData import BalanceData
from ds_charts import plot_evaluation_results, multiple_line_chart, get_variable_types

file_tag = 'knnOverfitting'
filename = '../data/encoded_scaled.csv'
target = 'PERSON_INJURY'

df = read_csv(f'{filename}')
print("shape: ", df.shape)

var = 'PERSON_INJURY'

killed = df.loc[df[var] == 'Killed']
injured = df.loc[df[var] == 'Injured']
injured = injured.sample(n=10000, random_state=90)
df = pd.concat([killed, injured], axis=0)
print("new shape: ", df.shape)

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

trnX, tstX, trnY, tstY = train_test_split(df, df['PERSON_INJURY'], test_size=0.3, random_state=1)
balanceData = BalanceData(dataframe=pd.concat([trnX], axis=1))
result_smote = balanceData.smote()
trnX = result_smote.copy()
trnY = result_smote['PERSON_INJURY']



for coiso in [(trnY, "train", trnX), (tstY, "test", tstX)]:

    figure()
    fig, axs = subplots(3, 1, figsize=(5, 5), dpi=150, squeeze=False)
    fig.tight_layout(pad=3.0)

    row = 0

    for score in [(accuracy_score, 'accuracy'), (precision_score, 'precision'), (recall_score, 'recall')]:

        nvalues = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
        dist = ['manhattan', 'euclidean', 'chebyshev']
        values = {}
        best = (0, '')
        last_best = 0
        for d in range(len(dist)):
            f = dist[d]
            yvalues = []
            for n in nvalues:
                knn = KNeighborsClassifier(n_neighbors=n, metric=dist[d])
                knn.fit(trnX, trnY)
                prdY = knn.predict(coiso[2])
                yvalues.append(score[0](coiso[0], prdY))

                if yvalues[-1] > last_best:
                    best = (n, d)
                    last_best = yvalues[-1]
            values[dist[d]] = yvalues

        multiple_line_chart(nvalues, values, ax=axs[row, 0],
                                title=f'KNN with {f} criteria - {score[1]} - @{coiso[1]}',
                                xlabel='n', ylabel="{}".format(score[1]), percentage=True)
        row += 1

    savefig(f'overfitting/{coiso[1]}_overfittingFor.png')

labels = unique(trnY)
labels.sort()
