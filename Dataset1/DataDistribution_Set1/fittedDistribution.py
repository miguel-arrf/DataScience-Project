import numpy as np
import pandas as pd
from matplotlib.pyplot import show, subplots, Axes, savefig
from numpy import log
from pandas import Series
from pandas.plotting import register_matplotlib_converters
from scipy.stats import norm, expon, lognorm

import ds_charts as ds

import os
from ds_charts import HEIGHT, multiple_line_chart, bar_chart, choose_grid, get_variable_types

register_matplotlib_converters()
currentPath = "/".join(os.path.abspath(__file__).split("/")[:-1]) 
filename =   '../../data/NYC_collisions_tabular.csv'
filename = "../teste_to_use.csv"


data = pd.read_csv(filename, index_col="UNIQUE_ID", na_values='', parse_dates=True, infer_datetime_format=True)
#data = data.drop(["VEHICLE_ID", "COLLISION_ID"], axis=1)

data = data.loc[(data['PERSON_AGE'] < 140) & (data['PERSON_AGE'] >= 0)]

def compute_known_distributions(x_values: list) -> dict:
    distributions = dict()
    # Gaussian
    print("Gaussian")
    mean, sigma = norm.fit(x_values)
    distributions['Normal(%.1f,%.2f)'%(mean,sigma)] = norm.pdf(x_values, mean, sigma)
    # Exponential
    print("Exponential")
    loc, scale = expon.fit(x_values)
    distributions['Exp(%.2f)'%(1/scale)] = expon.pdf(x_values, loc, scale)
    # LogNorm
    print("LogNormal")
    sigma, loc, scale = lognorm.fit(x_values)
    distributions['LogNor(%.1f,%.2f)'%(log(scale),sigma)] = lognorm.pdf(x_values, sigma, loc, scale)
    return distributions

def histogram_with_distributions(ax: Axes, series: Series, var: str):
    values = series.sort_values().values
    ax.hist(values, 20, density=True)
    distributions = compute_known_distributions(values)
    print("computer")
    multiple_line_chart(values, distributions, ax=ax, title='Best fit for %s'%var, xlabel=var, ylabel='')
    print("wut")

numeric_vars = get_variable_types(data)['Numeric']
print("numeric vars: ", numeric_vars)
if [] == numeric_vars:
    raise ValueError('There are no numeric variables.')

rows, cols = ds.choose_grid(len(numeric_vars))
fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
i, j = 0, 0
for n in range(len(numeric_vars)):
    histogram_with_distributions(axs[i, j], data[numeric_vars[n]].dropna(), numeric_vars[n])
    print("done")
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
savefig('images/histogram_numeric_distribution.png')

