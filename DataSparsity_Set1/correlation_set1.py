import pandas as pd
from matplotlib import pyplot as plt
from pandas import read_csv
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import savefig, show, title, figure
from seaborn import heatmap
import seaborn as sns

register_matplotlib_converters()
filename = '../data/NYC_collisions_tabular.csv'
data = read_csv(filename, index_col='CRASH_DATE', parse_dates=True, infer_datetime_format=True)

corr_mtx = data.corr()

fig = figure(figsize=[12, 12])

heatmap(abs(corr_mtx), xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=True, cmap='Blues')
title('Correlation analysis')
savefig(f'../DataSparsity_Set1/images/correlation_analysis.png')
show()

'''
# Basic correlogram
sns.pairplot(data, vars=data.columns[1:3])
savefig(f'../DataSparsity_Set1/images/teste.png')

show()
'''

features_to_use = ["BODILY_INJURY", "SAFETY_EQUIPMENT", "PERSON_SEX", "PED_LOCATION", "CONTRIBUTING_FACTOR_2",
                   "EJECTION", "COMPLAINT", "EMOTIONAL_STATUS", "CONTRIBUTING_FACTOR_1", "POSITION_IN_VEHICLE",
                   "PED_ACTION", "PERSON_INJURY"]

data = data[features_to_use]

for feature in features_to_use:
    data[feature] = data[feature].astype(str)

data['BODILY_INJURY'] = data.BODILY_INJURY.astype(str)
data['SAFETY_EQUIPMENT'] = data.SAFETY_EQUIPMENT.astype(str)
data['PERSON_SEX'] = data.PERSON_SEX.astype(str)

print(data.dtypes)


# TODO: This was taken from somewhere over stackoverflow, it is important to figure out from where and refer it on
#  the report! create customized scatterplot that first filters out NaNs in feature pair
def scatterFilter(x, y, **kwargs):
    interimDf = pd.concat([x, y], axis=1)
    interimDf.columns = ['x', 'y']
    interimDf = interimDf[(~ pd.isnull(interimDf.x)) & (~ pd.isnull(interimDf.y))]

    ax = plt.gca()
    # print(interimDf.x.values)
    ax = sns.stripplot(x=interimDf.x.values, y=interimDf.y.values, jitter=True, palette='viridis')
    # ax = plt.scatter(interimDf.x.values, interimDf.y.values, 'o', jitter=True, **kwargs)


# Create an instance of the PairGrid class.
grid = sns.PairGrid(data=data, vars=list(data.columns), height=4)

# Map a scatter plot to the upper triangle
grid = grid.map_upper(scatterFilter, color='darkred')

# Map a histogram to the diagonal
grid = grid.map_diag(plt.hist, bins=10, edgecolor='k', color='darkred')

# Map a density plot to the lower triangle
grid = grid.map_lower(scatterFilter, color='darkred')

savefig(f'../DataSparsity_Set1/images/sparsity_symbolic_relevant_variables.png')

plt.show()
