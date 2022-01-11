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

rows, cols = choose_grid(len(variables))
fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
i, j = 0, 0

for n in range(len(variables)):
    axs[i, j].set_title(f'Histogram for {variables[n]}')
    axs[i, j].set_xlabel(variables[n])
    axs[i, j].set_ylabel('nr records')
    axs[i, j].hist(data[variables[n]].values, bins=100)
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
savefig('../DataGranularity_Set1/images/granularity_single.png')
show()
