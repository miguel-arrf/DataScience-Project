import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure, show, subplots, Axes
from ds_charts import multiple_bar_chart, HEIGHT, multiple_line_chart, bar_chart, choose_grid, get_variable_types
import pandas as pd
from pandas.plotting import register_matplotlib_converters
import ds_charts as ds
from seaborn import distplot
from numpy import log
from pandas import Series
from scipy.stats import norm, expon, lognorm

import os

register_matplotlib_converters()

currentPath = "/".join(os.path.abspath(__file__).split("/")[:-1]) 
filename =  currentPath + '/../data/NYC_collisions_tabular.csv'

data = pd.read_csv(filename, index_col="COLLISION_ID", na_values='', parse_dates=True, infer_datetime_format=True)

NR_STDEV: int = 2


numeric_data = data.select_dtypes(include=np.number)
numeric_vars = numeric_data.columns.tolist()
if [] == numeric_vars:
    raise ValueError('There are no numeric variables.')

outliers_iqr = []
outliers_stdev = []
summary_5 = data.describe(include='number')

for var in numeric_vars:
    iqr = 1.5 * (summary_5[var].quantile(q=0.75) - summary_5[var].quantile(q=0.25))
    outliers_iqr += [
        data[data[var] > summary_5[var].quantile(q=0.75) + iqr].count()[var] +
        data[data[var] < summary_5[var].quantile(q=0.25) - iqr].count()[var]]
    std = NR_STDEV * summary_5[var].std()
    outliers_stdev += [
        data[data[var] > summary_5[var].mean() + std].count()[var] +
        data[data[var] < summary_5[var].mean() - std].count()[var]]

outliers = {'iqr': outliers_iqr, 'stdev': outliers_stdev}
figure(figsize=(12, HEIGHT))
multiple_bar_chart(numeric_vars, outliers, title='Nr of outliers per variable', xlabel='variables',
                   ylabel='nr outliers', percentage=False)
plt.savefig("images/outliers.png")

show()

print(outliers_iqr)
print("--")
print(outliers_stdev)

rows, cols = ds.choose_grid(len(numeric_vars))
print("rows: {}, cols: {}".format(rows, cols))
fig, axs = subplots(rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT))
i, j = 0, 0
for n in range(len(numeric_vars)):
    axs[i, j].set_title('Histogram for %s' % numeric_vars[n])
    axs[i, j].set_xlabel(numeric_vars[n])
    axs[i, j].set_ylabel("nr records")
    axs[i, j].hist(data[numeric_vars[n]].dropna().values, 'auto')
    i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
plt.savefig('images/single_histograms_numeric.png')
show()

fig, axs = subplots(rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT))
i, j = 0, 0
for n in range(len(numeric_vars)):
    
    axs[i, j].set_title('Histogram with trend for %s' % numeric_vars[n])
    distplot(data[numeric_vars[n]].dropna().values, norm_hist=True, ax=axs[i, j], axlabel=numeric_vars[n])
    i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
plt.savefig('/images/histograms_trend_numeric.png')
show()

