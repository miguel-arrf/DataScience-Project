import numpy as np
import pandas as pd
from matplotlib.pyplot import show, subplots, Axes, savefig
from numpy import log
from pandas import Series
from pandas.plotting import register_matplotlib_converters
from scipy.stats import norm, expon, lognorm

import ds_charts as ds
from ds_charts import HEIGHT, multiple_line_chart, bar_chart, choose_grid, get_variable_types

register_matplotlib_converters()
filename = '../data/air_quality_tabular.csv'
data = pd.read_csv(filename, index_col='date', na_values='', parse_dates=True, infer_datetime_format=True)

data = data[ ["CO_Min", "CO_Max", "NO2_Min", "NO2_Max"]]
data = np.array_split(data, 6)[0]
print(data.describe())
print(data.shape)
print("head:")
print(data.head())

# Count total NaN at each column in a DataFrame
print(" \nCount total NaN at each column in a DataFrame : \n\n",
      data.isnull().sum())

NR_STDEV: int = 2

# data = data.drop('FID', axis=1)

features_to_use = ["CO_Min", "CO_Max", "NO2_Min", "NO2_Max"]
# "O3_Min", "O3_Max", "PM2.5_Min", "PM2.5_Max","PM10_Min", "PM10_Max", "SO2_Min", "SO2_Max"]


numeric_data = data.select_dtypes(include=np.number)
numeric_vars = numeric_data.columns.tolist()
numeric_vars = features_to_use
rows, cols = ds.choose_grid(len(numeric_vars))
print("Numeric vars: ", numeric_vars)


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
    print("var: ", numeric_vars[n])
    histogram_with_distributions(axs[i, j], data[numeric_vars[n]].dropna(), numeric_vars[n])
    i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
savefig(f'../DataDistribution_Set2/images/best_fit_for_valuable_features.png')
show()

