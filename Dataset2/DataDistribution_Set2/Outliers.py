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

register_matplotlib_converters()
filename = '../data/air_quality_tabular.csv'
data = pd.read_csv(filename, index_col='FID', na_values='', parse_dates=True, infer_datetime_format=True)

NR_STDEV: int = 2

#data = data.drop('FID', axis=1)

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
plt.savefig(f'../DataDistribution_Set2/images/outliers.png')

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
plt.savefig(f'../DataDistribution_Set2/images/single_histograms_numeric.png')
show()

fig, axs = subplots(rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT))
i, j = 0, 0
for n in range(len(numeric_vars)):
    axs[i, j].set_title('Histogram with trend for %s' % numeric_vars[n])
    distplot(data[numeric_vars[n]].dropna().values, norm_hist=True, ax=axs[i, j], axlabel=numeric_vars[n])
    i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
plt.savefig(f'../DataDistribution_Set2/images/histograms_trend_numeric.png')
show()



'''

def compute_known_distributions(x_values: list) -> dict:
    distributions = dict()
    # Gaussian
    mean, sigma = norm.fit(x_values)
    distributions['Normal(%.1f,%.2f)' % (mean, sigma)] = norm.pdf(x_values, mean, sigma)
    # Exponential
    loc, scale = expon.fit(x_values)
    distributions['Exp(%.2f)' % (1 / scale)] = expon.pdf(x_values, loc, scale)
    # LogNorm
    sigma, loc, scale = lognorm.fit(x_values)
    distributions['LogNor(%.1f,%.2f)' % (log(scale), sigma)] = lognorm.pdf(x_values, sigma, loc, scale)
    return distributions


def histogram_with_distributions(ax: Axes, series: Series, var: str):
    values = series.sort_values().values
    ax.hist(values, 20, density=True)
    distributions = compute_known_distributions(values)
    multiple_line_chart(values, distributions, ax=ax, title='Best fit for %s' % var, xlabel=var, ylabel='')


fig, axs = subplots(rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT))
i, j = 0, 0
for n in range(len(numeric_vars)):
    histogram_with_distributions(axs[i, j], data[numeric_vars[n]].dropna(), numeric_vars[n])
    i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
show()




symbolic_vars = get_variable_types(data)['Symbolic']
if [] == symbolic_vars:
    raise ValueError('There are no symbolic variables.')

rows, cols = choose_grid(len(symbolic_vars))
fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
i, j = 0, 0
for n in range(len(symbolic_vars)):
    counts = data[symbolic_vars[n]].value_counts()
    bar_chart(counts.index.to_list(), counts.values, ax=axs[i, j], title='Histogram for %s'%symbolic_vars[n], xlabel=symbolic_vars[n], ylabel='nr records', percentage=False)
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
show()

'''
