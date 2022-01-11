from pandas import read_csv
from ds_charts import get_variable_types, choose_grid, HEIGHT
from matplotlib.pyplot import subplots, savefig, show

filename =   '../../data/NYC_collisions_tabular.csv'
data = read_csv(filename)
data = data.drop(["VEHICLE_ID", "COLLISION_ID", "PERSON_ID", "UNIQUE_ID"], axis=1)
data = data.loc[(data['PERSON_AGE'] < 140) & (data['PERSON_AGE'] >= 0)]
values = {'nr records': data.shape[0], 'nr variables': data.shape[1]}

variables = get_variable_types(data)['Numeric']
if [] == variables:
    raise ValueError('There are no numeric variables.')

rows = len(variables)
bins = (5, 10, 15, 20, 50)
cols = len(bins)
fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)

for i in range(rows):
    for j in range(cols):
        axs[i, j].set_title(f'Histogram for {variables[i]} {bins[j]} bins')
        axs[i, j].set_xlabel(variables[i])
        axs[i, j].set_ylabel('Nr records')
        axs[i, j].hist(data[variables[i]].values, bins=bins[j])
savefig('../DataGranularity_Set1/images/granularity_study.png')
show()