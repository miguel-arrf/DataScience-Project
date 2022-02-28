import pandas as pd
from pandas import DataFrame, read_csv
from matplotlib.pyplot import figure, xlabel, ylabel, scatter, show, subplots, savefig
from sklearn.decomposition import PCA
from numpy.linalg import eig
from matplotlib.pyplot import gca, title
from sklearn.model_selection import train_test_split

from ds_charts import get_variable_types

target = 'ALARM'

data = read_csv(f'../../data/air_quality_tabular.csv')
data = data.dropna()
symbolic_vars = get_variable_types(data)['Symbolic']
for symbolic_var in symbolic_vars:
    data[symbolic_var] = pd.factorize(data[symbolic_var])[0]

binary_vars = get_variable_types(data)['Binary']
for binary_var in binary_vars:
    data[binary_var] = pd.factorize(data[binary_var])[0]
y = data['ALARM']
df = data.drop('ALARM', 1)

print("columns: ", data.columns)

variables = data.columns.values
eixo_x = 20  # pm2.5_mean
eixo_y = 24  # pm10_mean
eixo_z = 28  # s02_mean

eixo_x = 0
eixo_y = 1
eixo_z = 2

data = data[['CO_Mean', 'CO_Min', 'CO_Max', 'CO_Std', 'NO2_Mean', 'NO2_Min',
       'NO2_Max', 'NO2_Std', 'O3_Mean', 'O3_Min', 'O3_Max', 'O3_Std',
       'PM2.5_Mean', 'PM2.5_Min', 'PM2.5_Max', 'PM2.5_Std', 'PM10_Mean',
       'PM10_Min', 'PM10_Max', 'PM10_Std', 'SO2_Mean', 'SO2_Min', 'SO2_Max',
       'SO2_Std']]

data = data[['CO_Mean', 'CO_Min', 'CO_Max', 'CO_Std']]

print("columns: ", data.columns)

figure()
xlabel(variables[eixo_y])
ylabel(variables[eixo_z])
scatter(data.iloc[:, eixo_y], data.iloc[:, eixo_z])
savefig(f"images/FeatureExtraction")
# show()

mean = (data.mean(axis=0)).tolist()
centered_data = data - mean
cov_mtx = centered_data.cov()
eigvals, eigvecs = eig(cov_mtx)

pca = PCA()
pca.fit(centered_data)
PC = pca.components_
var = pca.explained_variance_

# PLOT EXPLAINED VARIANCE RATIO
fig = figure(figsize=(4, 4))
title('Explained variance ratio')
xlabel('PCA')
ylabel('ratio')
x_values = [str(i) for i in range(1, len(pca.components_) + 1)]
bwidth = 0.5
ax = gca()
ax.set_xticklabels(x_values)
ax.set_ylim(0.0, 1.0)
ax.bar(x_values, pca.explained_variance_ratio_, width=bwidth)
ax.plot(pca.explained_variance_ratio_)
for i, v in enumerate(pca.explained_variance_ratio_):
    ax.text(i, v + 0.05, f'{v * 100:.1f}', ha='center', fontweight='bold')
savefig(f"images/ExplainedVarianceRatio")
# show()

transf = pca.transform(data)

_, axs = subplots(1, 2, figsize=(2 * 5, 1 * 5), squeeze=False)
axs[0, 0].set_xlabel(variables[eixo_y])
axs[0, 0].set_ylabel(variables[eixo_z])
axs[0, 0].scatter(data.iloc[:, eixo_y], data.iloc[:, eixo_z])

axs[0, 1].set_xlabel('PC1')
axs[0, 1].set_ylabel('PC2')
axs[0, 1].scatter(transf[:, 0], transf[:, 1])
savefig(f"images/PCA")
# show()
