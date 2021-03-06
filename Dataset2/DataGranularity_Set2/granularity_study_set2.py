import matplotlib.pyplot as plt
from pandas import read_csv
from ds_charts import get_variable_types, choose_grid, HEIGHT
from matplotlib.pyplot import subplots, savefig, show
import os

currentPath = "/".join(os.path.abspath(__file__).split("/")[:-1]) 
filename =  '../../data/air_quality_tabular.csv'

data = read_csv(filename)

values = {'nr records': data.shape[0], 'nr variables': data.shape[1]}

variables = get_variable_types(data)['Numeric']
if [] == variables:
    raise ValueError('There are no numeric variables.')

rows = len(variables)
bins = (10, 100, 1000)
cols = len(bins)
fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)

for i in range(rows):
    for j in range(cols):
        axs[i, j].set_title(f'Histogram for {variables[i]} {bins[j]} bins')
        axs[i, j].set_xlabel(variables[i])
        axs[i, j].set_ylabel('Nr records')
        axs[i, j].hist(data[variables[i]].values, bins=bins[j])

plt.tight_layout()
savefig(currentPath + '/images/granularity_study.png', dpi=150)
show()