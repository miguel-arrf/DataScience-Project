from matplotlib import pyplot as plt
from numpy import ndarray
from pandas import DataFrame, read_csv, unique, factorize
from matplotlib.pyplot import figure, subplots, savefig, show
from sklearn.neural_network import MLPClassifier
from ds_charts import plot_evaluation_results, multiple_line_chart, horizontal_bar_chart, HEIGHT, get_variable_types
from sklearn.metrics import accuracy_score
from ds_charts import plot_overfitting_study
from sklearn.model_selection import train_test_split

file_tag = 'air_quality'
target = 'ALARM'

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

trnX, tstX, trnY, tstY = train_test_split(df, df[target], test_size=0.3, random_state=1,
                                          stratify=df[target])
labels = unique(trnY)
labels.sort()

mlp = MLPClassifier(activation='logistic', solver='sgd', learning_rate='adaptive',
                    learning_rate_init=0.01, max_iter=300, verbose=False)
mlp.fit(trnX, trnY)

prd_trn = mlp.predict(trnX)
prd_tst = mlp.predict(tstX)
#plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst)
# savefig(f'images/{file_tag}_mlp_best.png')
test_acc = accuracy_score(tstY, prd_tst) * 100.

loss_values = mlp.loss_curve_

plt.plot(loss_values)
plt.show()