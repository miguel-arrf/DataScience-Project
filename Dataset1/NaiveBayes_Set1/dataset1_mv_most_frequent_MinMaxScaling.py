import matplotlib.pyplot as plt
from pandas import read_csv
from pandas.plotting import register_matplotlib_converters
from ds_charts import get_variable_types
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame, concat
from matplotlib.pyplot import subplots, show

register_matplotlib_converters()
filename = '../MissingValuesImputation_Set1/data/dataset1_mv_most_frequent.csv'
data = read_csv(filename, parse_dates=True, infer_datetime_format=True)

variable_types = get_variable_types(data)
numeric_vars = variable_types['Numeric']
symbolic_vars = variable_types['Symbolic']
boolean_vars = variable_types['Binary']

df_nr = data[numeric_vars]
df_sb = data[symbolic_vars]
df_bool = data[boolean_vars]


transf = MinMaxScaler(feature_range=(0, 1), copy=True).fit(df_nr)
tmp = DataFrame(transf.transform(df_nr), index=data.index, columns=numeric_vars)
norm_data_minmax = concat([tmp, df_sb, df_bool], axis=1)
norm_data_minmax.to_csv('minMaxScaling.csv', index=False)


