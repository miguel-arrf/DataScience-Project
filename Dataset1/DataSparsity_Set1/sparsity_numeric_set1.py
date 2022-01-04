import numpy as np
from matplotlib.pyplot import subplots, savefig, show
from pandas import read_csv
from pandas.plotting import register_matplotlib_converters
from ds_charts import get_variable_types, HEIGHT

register_matplotlib_converters()
filename = '../Dataset1/data/grouped.csv'
data = read_csv(filename, index_col='UNIQUE_ID', parse_dates=True, infer_datetime_format=True)

numeric_vars = get_variable_types(data)['Numeric']
numeric_vars.append('PERSON_INJURY')
# symbolic_vars = ['PERSON_INJURY']

print("Numeric vars: ", numeric_vars)

if not numeric_vars:
    raise ValueError('There are no numeric variables.')

rows, cols = len(numeric_vars) - 1, len(numeric_vars) - 1
fig, axs = subplots(rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False)

colors = np.where(data["PERSON_INJURY"] == "Injured", 'y', 'k')

for i in range(len(numeric_vars)):
    var1 = numeric_vars[i]
    for j in range(i + 1, len(numeric_vars)):
        var2 = numeric_vars[j]
        axs[i, j - 1].set_title("%s x %s" % (var1, var2))
        axs[i, j - 1].set_xlabel(var1)
        axs[i, j - 1].set_ylabel(var2)
        axs[i, j - 1].scatter(data[var1], data[var2], c=colors, cmap='winter')
savefig(f'../DataSparsity_Set1/images/sparsity_study_numeric.png')
show()
