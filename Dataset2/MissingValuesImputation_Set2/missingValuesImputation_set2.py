import pandas as pd
import os

from matplotlib.pyplot import savefig, show, title, figure
from sklearn.impute import SimpleImputer

from pandas import read_csv, concat, DataFrame
from pandas.plotting import register_matplotlib_converters

from seaborn import heatmap
from ds_charts import get_variable_types
from numpy import nan


pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 20)


register_matplotlib_converters()

currentPath = "/".join(os.path.abspath(__file__).split("/")[:-1]) 
filename =  currentPath + '/../data/air_quality_tabular.csv'

data = read_csv(filename, index_col='FID', parse_dates=True, infer_datetime_format=True)

# savefig(f'../DataCorrelation_Set1/images/correlation_analysis.png')
print(data.shape)
mv = {}
figure()
for var in data:
    nr = data[var].isna().sum()
    if nr > 0:
        mv[var] = nr

print(mv)

# defines the number of records to discard entire columns
threshold = data.shape[0] * 0.90

missings = [c for c in mv.keys() if mv[c] > threshold]
df = data.drop(columns=missings, inplace=False)
df.to_csv(currentPath + '/data/dataset2_drop_columns_mv.csv', index=False)
print('Dropped variables', missings)

# defines the number of variables to discard entire records
threshold = data.shape[1] * 0.50

df = data.dropna(thresh=threshold, inplace=False)
df.to_csv(currentPath + '/data/dataset2_drop_records_mv.csv', index=False)
print(df.shape)




tmp_nr, tmp_sb, tmp_bool = None, None, None
variables = get_variable_types(data)
numeric_vars = variables['Numeric']
symbolic_vars = variables['Symbolic']
binary_vars = variables['Binary']

if len(numeric_vars) > 0:
    imp = SimpleImputer(strategy='constant', fill_value=0, missing_values=nan, copy=True)
    tmp_nr = DataFrame(imp.fit_transform(data[numeric_vars]), columns=numeric_vars)
if len(symbolic_vars) > 0:
    imp = SimpleImputer(strategy='constant', fill_value='NA', missing_values=nan, copy=True)
    tmp_sb = DataFrame(imp.fit_transform(data[symbolic_vars]), columns=symbolic_vars)
if len(binary_vars) > 0:
    imp = SimpleImputer(strategy='constant', fill_value=False, missing_values=nan, copy=True)
    tmp_bool = DataFrame(imp.fit_transform(data[binary_vars]), columns=binary_vars)

df = concat([tmp_nr, tmp_sb, tmp_bool], axis=1)
df.to_csv(currentPath + '/data/dataset2_mv_constant.csv', index=False)

tmp_nr, tmp_sb, tmp_bool = None, None, None
variables = get_variable_types(data)
numeric_vars = variables['Numeric']
symbolic_vars = variables['Symbolic']
binary_vars = variables['Binary']

tmp_nr, tmp_sb, tmp_bool = None, None, None
if len(numeric_vars) > 0:
    imp = SimpleImputer(strategy='mean', missing_values=nan, copy=True)
    tmp_nr = DataFrame(imp.fit_transform(data[numeric_vars]), columns=numeric_vars)
if len(symbolic_vars) > 0:
    imp = SimpleImputer(strategy='most_frequent', missing_values=nan, copy=True)
    tmp_sb = DataFrame(imp.fit_transform(data[symbolic_vars]), columns=symbolic_vars)
if len(binary_vars) > 0:
    imp = SimpleImputer(strategy='most_frequent', missing_values=nan, copy=True)
    tmp_bool = DataFrame(imp.fit_transform(data[binary_vars]), columns=binary_vars)

df = concat([tmp_nr, tmp_sb, tmp_bool], axis=1)
df.to_csv(currentPath + '/data/dataset2_mv_most_frequent.csv', index=False)


