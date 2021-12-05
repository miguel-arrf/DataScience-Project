from matplotlib.pyplot import savefig, show, title, figure
from pandas import read_csv
from pandas.plotting import register_matplotlib_converters
from seaborn import heatmap

register_matplotlib_converters()
filename = '../data/NYC_collisions_tabular.csv'
data = read_csv(filename, index_col='UNIQUE_ID', parse_dates=True, infer_datetime_format=True)


fig = figure(figsize=[12, 12])

corr_mtx = abs(data.corr())

heatmap(abs(corr_mtx), xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=True, cmap='Blues')
title('Correlation analysis')
savefig(f'../DataCorrelation_Set1/images/correlation_analysis.png')
show()