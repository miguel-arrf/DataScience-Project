from matplotlib.pyplot import savefig, show, title, figure
from pandas import read_csv
from pandas.plotting import register_matplotlib_converters
from seaborn import heatmap

register_matplotlib_converters()
filename = '../data/air_quality_tabular.csv'
data = read_csv(filename, index_col='FID', parse_dates=True, infer_datetime_format=True)


fig = figure(figsize=[22, 22])

corr_mtx = abs(data.corr())

heatmap(abs(corr_mtx), xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=True, cmap='Blues')
title('Correlation analysis')
savefig(f'../DataCorrelation_Set2/images/correlation_analysis.png')
show()