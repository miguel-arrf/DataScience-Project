import pandas as pd
from matplotlib.pyplot import savefig, show, title, figure
from pandas import read_csv
from pandas.plotting import register_matplotlib_converters
from seaborn import heatmap
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 20)


register_matplotlib_converters()
filename = '../data/NYC_collisions_tabular.csv'
data = read_csv(filename, index_col='UNIQUE_ID', parse_dates=True, infer_datetime_format=True)

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
df.to_csv(f'data/dataset1_drop_columns_mv.csv', index=False)
print('Dropped variables', missings)

# defines the number of variables to discard entire records
threshold = data.shape[1] * 0.50

df = data.dropna(thresh=threshold, inplace=False)
df.to_csv(f'data/dataset1_drop_records_mv.csv', index=False)
print(df.shape)

# TODO: Teacher talks about the 'IterativeImputer', which looks quite promising.
#  On this tutorial she only uses simple methods tho...

columns_to_check_effect = ["PERSON_AGE", "SAFETY_EQUIPMENT", "PED_LOCATION", "CONTRIBUTING_FACTOR_2", "EJECTION",
                           "VEHICLE_ID", "CONTRIBUTING_FACTOR_1", "POSITION_IN_VEHICLE","PED_ACTION"]

from sklearn.impute import SimpleImputer
from pandas import concat, DataFrame
from ds_charts import get_variable_types
from numpy import nan

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
df.to_csv(f'data/dataset1_mv_constant.csv', index=False)
print(df[columns_to_check_effect].describe(include='all'))




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
df.to_csv(f'data/dataset1_mv_most_frequent.csv', index=False)
print(df[columns_to_check_effect].describe(include='all'))


# TODO: For the ID's columns it doesn't make sense for us to be doing anything... I think it is irrelevant what we
#  put there since we aren't going to use it. If we end up using the ID's columns, we should put something unique
#  even though, for VEHICLE_ID, the same car can be involved in multiple accidents.

# TODO: For some variables the mean or wtv value makes sense, for others it doesn't (PED_LOCATION, FACTOR_2 and 1 and
#  PED_ACTION).
