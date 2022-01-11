import pandas as pd
from matplotlib.pyplot import savefig, show, title, figure
from pandas import read_csv
from pandas.plotting import register_matplotlib_converters
from seaborn import heatmap

from ds_charts import get_variable_types

register_matplotlib_converters()
filename =   '../../data/NYC_collisions_tabular.csv'
data = read_csv(filename)
data = data.drop(["VEHICLE_ID", "COLLISION_ID", "PERSON_ID", "UNIQUE_ID"], axis=1)
data = data.loc[(data['PERSON_AGE'] < 140) & (data['PERSON_AGE'] >= 0)]


data = data.loc[data["PERSON_INJURY"] == 'Killed']


data['PERSON_INJURY'] = pd.factorize(data['PERSON_INJURY'])[0]
fig = figure(figsize=[22, 22])


corr_mtx = abs(data.corr())
'''
heatmap(abs(corr_mtx), xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=True, cmap='Blues')
title('Correlation analysis')
savefig(f'../DataCorrelation_Set1/images/correlation_analysis_numericWithTarget.png')
show()
'''

fig = figure(figsize=[22, 22])
symbolic_vars = get_variable_types(data)['Symbolic']
for symbolic_var in symbolic_vars:
    data[symbolic_var] = pd.factorize(data[symbolic_var])[0]

good_columns = []
for column in data.columns:
    if "_ID" not in column:
        good_columns.append(column)

data = data[good_columns]


corr_mtx = abs(data.corr())
import seaborn as sns
sns.set(font_scale=1.1)
#sns.color_palette("rocket", as_cmap=True)
#, cmap='Blues'
heatmap(abs(corr_mtx), xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=True,  cmap='Blues')
title('Correlation analysis')
savefig(f'../DataCorrelation_Set1/images/correlationKilled.png')
show()