import numpy as np
from pandas import read_csv
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import savefig, show, subplots
from ds_charts import HEIGHT, get_variable_types

register_matplotlib_converters()
filename = '../data/air_quality_tabular.csv'
data = read_csv(filename, index_col='FID', parse_dates=True, infer_datetime_format=True)

symbolic_vars = get_variable_types(data)['Symbolic']
if [] == symbolic_vars:
    raise ValueError('There are no symbolic variables.')

rows, cols = len(symbolic_vars)-1, len(symbolic_vars)-1
fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)

colors = np.where(data["ALARM"] == "Safe", 'y', 'k')


for i in range(len(symbolic_vars)):
    var1 = symbolic_vars[i]
    for j in range(i+1, len(symbolic_vars)):
        var2 = symbolic_vars[j]
        axs[i, j-1].set_title("%s x %s"%(var1,var2))
        axs[i, j-1].set_xlabel(var1)
        axs[i, j-1].set_ylabel(var2)
        axs[i, j-1].scatter(data[var1].astype(str), data[var2].astype(str),  c=colors)
savefig(f'../DataSparsity_Set2/images/sparsity_study_symbolic.png')
show()