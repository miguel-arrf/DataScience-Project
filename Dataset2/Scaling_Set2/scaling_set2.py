import matplotlib.pyplot as plt
from pandas import read_csv
from pandas.plotting import register_matplotlib_converters
from ds_charts import get_variable_types
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame, concat
from matplotlib.pyplot import subplots, show

register_matplotlib_converters()
filename = '../data/air_quality_tabular.csv'
data = read_csv(filename, index_col='FID', parse_dates=True, infer_datetime_format=True)

variable_types = get_variable_types(data)
numeric_vars = variable_types['Numeric']
symbolic_vars = variable_types['Symbolic']
boolean_vars = variable_types['Binary']

df_nr = data[numeric_vars]
df_sb = data[symbolic_vars]
df_bool = data[boolean_vars]

transf = StandardScaler(with_mean=True, with_std=True, copy=True).fit(df_nr)
tmp = DataFrame(transf.transform(df_nr), index=data.index, columns=numeric_vars)
norm_data_zscore = concat([tmp, df_sb, df_bool], axis=1)
norm_data_zscore.to_csv(f'data/air_quality_tabular_scaled_zscore.csv', index=False)

transf = MinMaxScaler(feature_range=(0, 1), copy=True).fit(df_nr)
tmp = DataFrame(transf.transform(df_nr), index=data.index, columns=numeric_vars)
norm_data_minmax = concat([tmp, df_sb, df_bool], axis=1)
norm_data_minmax.to_csv(f'data/air_quality_tabular_scaled_minmax.csv', index=False)


fig, axs = subplots(1, 3, figsize=(30,20),squeeze=False)
axs[0, 0].set_title('Original data')
data.boxplot(ax=axs[0, 0])
axs[0, 1].set_title('Z-score normalization')
norm_data_zscore.boxplot(ax=axs[0, 1])
axs[0, 2].set_title('MinMax normalization')
norm_data_minmax.boxplot(ax=axs[0, 2])

for tick in axs[0, 0].get_xticklabels():
    tick.set_rotation(90)

for tick in axs[0, 1].get_xticklabels():
    tick.set_rotation(90)

for tick in axs[0, 2].get_xticklabels():
    tick.set_rotation(90)

plt.savefig(f'../Scaling_Set2/images/scaling.png')
show()

