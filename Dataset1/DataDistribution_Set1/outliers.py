import os
from collections import Counter

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure, show, subplots
from pandas.plotting import register_matplotlib_converters
from seaborn import distplot

import ds_charts as ds
from ds_charts import multiple_bar_chart, HEIGHT

register_matplotlib_converters()

currentPath = "/".join(os.path.abspath(__file__).split("/")[:-1])
filetag = "encoded_notScaled"
filename =   '../../data/NYC_collisions_tabular.csv'
filename = "../teste_to_use.csv"


data = pd.read_csv(filename, index_col="UNIQUE_ID", na_values='', parse_dates=True, infer_datetime_format=True)
#data = data.drop(["VEHICLE_ID", "COLLISION_ID"], axis=1)

#data = data.loc[(data['PERSON_AGE'] < 140) & (data['PERSON_AGE'] >= 0)]

from matplotlib.pyplot import figure, savefig, show
from ds_charts import get_variable_types, multiple_bar_chart, HEIGHT

NR_STDEV: int = 2

numeric_vars = get_variable_types(data)['Numeric']
print("numeric vars: ", numeric_vars)
if [] == numeric_vars:
    raise ValueError('There are no numeric variables.')

outliers_iqr = []
outliers_stdev = []
summary5 = data.describe(include='number')

for var in numeric_vars:
    iqr = 1.5 * (summary5[var]['75%'] - summary5[var]['25%'])
    outliers_iqr += [
        data[data[var] > summary5[var]['75%'] + iqr].count()[var] +
        data[data[var] < summary5[var]['25%'] - iqr].count()[var]]
    std = NR_STDEV * summary5[var]['std']
    outliers_stdev += [
        data[data[var] > summary5[var]['mean'] + std].count()[var] +
        data[data[var] < summary5[var]['mean'] - std].count()[var]]

    if var == "PERSON_AGE":
        tempVar = "PERSON_INJURY"

        print("IQR:")
        print(">75% :")
        print(Counter(data[data[var] > summary5[var]['75%'] + iqr][tempVar].tolist()))
        print("<75% :")
        print(Counter(data[data[var] < summary5[var]['25%'] - iqr][tempVar].tolist()))
        print("--------")
        print("stdev:")
        print("+std :")
        print(Counter(data[data[var] > summary5[var]['mean'] + std][tempVar].tolist()))
        print("-std :")
        print(Counter(data[data[var] < summary5[var]['mean'] - std][tempVar].tolist()))

outliers = {'iqr': outliers_iqr, 'stdev': outliers_stdev}


figure(figsize=(12, HEIGHT))
multiple_bar_chart(numeric_vars, outliers, title='Nr of outliers per variable', xlabel='variables',
                   ylabel='nr outliers', percentage=False)
savefig('images/outliers_{}.png'.format(filetag))
show()


'''
rows, cols = ds.choose_grid(len(numeric_vars))
print("rows: {}, cols: {}".format(rows, cols))
fig, axs = subplots(rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False)
i, j = 0, 0
for n in range(len(numeric_vars)):
    axs[i, j].set_title('Histogram for %s' % numeric_vars[n])
    axs[i, j].set_xlabel(numeric_vars[n])
    axs[i, j].set_ylabel("nr records")
    axs[i, j].hist(data[numeric_vars[n]].dropna().values, 'auto')
    i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
plt.savefig('images/single_histograms_numeric_{}.png'.format(filetag))
show()

fig, axs = subplots(rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False)
i, j = 0, 0
for n in range(len(numeric_vars)):
    axs[i, j].set_title('Histogram with trend for %s' % numeric_vars[n])
    distplot(data[numeric_vars[n]].dropna().values, norm_hist=True, ax=axs[i, j], axlabel=numeric_vars[n])
    i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
plt.savefig('images/histograms_trend_numeric_{}.png'.format(filetag))
show()
'''
