import pandas as pd
from matplotlib.pyplot import show, subplots, savefig
from pandas.plotting import register_matplotlib_converters

from ds_charts import HEIGHT, bar_chart, choose_grid, get_variable_types

register_matplotlib_converters()
filename = '../data/air_quality_tabular.csv'
data = pd.read_csv(filename, index_col='FID', na_values='', parse_dates=True, infer_datetime_format=True)

symbolic_vars = get_variable_types(data)['Symbolic']
print("symbolic_vars: ", symbolic_vars)
if not symbolic_vars:
    raise ValueError('There are no symbolic variables.')


rows, cols = choose_grid(len(symbolic_vars))
fig, axs = subplots(rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False)
i, j = 0, 0
for n in range(len(symbolic_vars)):
    counts = data[symbolic_vars[n]].value_counts()
    new_counts_index = []
    for index in counts.index.to_list():
        new_counts_index.append(str(index))
    bar_chart(new_counts_index, counts.values, ax=axs[i, j], title='Histogram for %s' % symbolic_vars[n],
              xlabel=symbolic_vars[n], ylabel='nr records', percentage=False)
    i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
savefig(f'../DataDistribution_Set2/images/histograms_symbolic.png')
show()


rows, cols = choose_grid(len(symbolic_vars))
fig, axs = subplots(rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False)
i, j = 0, 0
for n in range(len(symbolic_vars)):
    counts = data[symbolic_vars[n]].value_counts()
    new_counts_index = []
    for index in counts.index.to_list():
        new_counts_index.append(str(index))

    bar_chart(new_counts_index, counts.values/sum(counts.values), ax=axs[i, j], title='Histogram for %s' % symbolic_vars[n],
              xlabel=symbolic_vars[n], ylabel='nr records', percentage=True)
    i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
savefig(f'../DataDistribution_Set2/images/class_distribution_symbolic_max_as_one.png')
show()

# TODO: I think this function is doing nothing different...
'''
rows, cols = choose_grid(len(symbolic_vars))
fig, axs = subplots(rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False)
i, j = 0, 0
for n in range(len(symbolic_vars)):
    counts = data[symbolic_vars[n]].value_counts()
    new_counts_index = []
    for index in counts.index.to_list():
        new_counts_index.append(str(index))

    bar_chart(new_counts_index, counts.values/sum(counts.values), ax=axs[i, j], title='Histogram for %s' % symbolic_vars[n],
              xlabel=symbolic_vars[n], ylabel='nr records', percentage=False)
    i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
savefig(f'../DataDistribution_Set2/images/class_distribution_symbolic.png')
show()'''