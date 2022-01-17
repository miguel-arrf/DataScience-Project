import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import read_csv
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import savefig, show, title, figure
from seaborn import heatmap
import seaborn as sns

register_matplotlib_converters()
filename = '../../data/NYC_collisions_tabular.csv'
data = read_csv(filename, index_col='UNIQUE_ID', parse_dates=True, infer_datetime_format=True)

features_to_use = ["BODILY_INJURY", "SAFETY_EQUIPMENT", "PERSON_SEX", "PED_LOCATION", "CONTRIBUTING_FACTOR_2",
                   "EJECTION", "COMPLAINT", "EMOTIONAL_STATUS", "CONTRIBUTING_FACTOR_1", "POSITION_IN_VEHICLE",
                   "PED_ACTION", "PERSON_INJURY"]

data = data[features_to_use]

for feature in features_to_use:
    data[feature] = data[feature].astype(str)

data['BODILY_INJURY'] = data.BODILY_INJURY.astype(str)
data['SAFETY_EQUIPMENT'] = data.SAFETY_EQUIPMENT.astype(str)
data['PERSON_SEX'] = data.PERSON_SEX.astype(str)

colors = np.where(data["PERSON_INJURY"] == "Injured", 'y', 'k')


# TODO: This was taken from somewhere over stackoverflow, it is important to figure out from where and refer it on
#  the report! create customized scatterplot that first filters out NaNs in feature pair
def scatterFilter(x, y, **kwargs):
    interimDf = pd.concat([x, y], axis=1)
    print("-> ", interimDf.columns)
    interimDf.columns = ['x', 'y']
    interimDf = interimDf[(~ pd.isnull(interimDf.x)) & (~ pd.isnull(interimDf.y))]

    ax = plt.gca()
    # print(interimDf.x.values)
    ax = sns.stripplot(x=interimDf.x.values, y=interimDf.y.values, jitter=True, palette='viridis')
    # ax = plt.scatter(interimDf.x.values, interimDf.y.values, 'o', jitter=True, **kwargs)


fig = plt.figure(figsize=(40,40))
fig.subplots_adjust(hspace=0.4, wspace=0.4)



for j in range(1, len(data.columns)):
    ax = fig.add_subplot(4, 3, j)
    sns.stripplot(data=data, x="PERSON_INJURY", y=data.columns[j], hue="PERSON_INJURY", jitter=True, palette='viridis')

savefig(f'../DataSparsity_Set1/images/sparsity_symbolic_relevant_variables.png')

show()
