import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import ds_charts as ds
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import savefig, show


register_matplotlib_converters()
filename = '../data/air_quality_tabular.csv'
data = pd.read_csv(filename, index_col='date', na_values='', parse_dates=True, infer_datetime_format=True)


data.boxplot(rot=45)
savefig(f'../DataDistribution_Set2/images/global_boxplot.png')
show()

print(data.isna().sum() / (len(data)) * 100)  # Relative missing values
print("--")
print(data.isna().sum())  # Absolute missing values

# Printing box plots for each numeric variable
numeric_data = data.select_dtypes(include=np.number)
print(numeric_data.columns.tolist())

rows, cols = ds.choose_grid(len(numeric_data.columns.tolist()))
fig, axs = plt.subplots(rows, cols, figsize=(cols * ds.HEIGHT, rows * ds.HEIGHT))
i, j = 0, 0

for n in range(len(numeric_data.columns.tolist())):
    print("for: {}".format(numeric_data.columns.tolist()[n]))
    axs[i, j].set_title('Boxplot for %s' % numeric_data.columns.tolist()[n])
    boxprops = dict(linestyle='-', linewidth=1, color='#005493')
    axs[i, j].boxplot(data[numeric_data.columns.tolist()[n]].dropna().values, boxprops=boxprops)

    i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
plt.savefig(f'../DataDistribution_Set2/images/single_boxplots.png')
plt.show()
