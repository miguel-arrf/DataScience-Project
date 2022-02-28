import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import ds_charts as ds
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import savefig, show

import os

register_matplotlib_converters()

currentPath = "/".join(os.path.abspath(__file__).split("/")[:-1])
#filename =  currentPath + '/../Dataset1/data/encoded_notScaled.csv'
#filename =   '../../data/NYC_collisions_tabular.csv'
filename = "../teste_to_use.csv"

data = pd.read_csv(filename, index_col="UNIQUE_ID", na_values='', parse_dates=True, infer_datetime_format=True)
#data = data.drop(["VEHICLE_ID", "COLLISION_ID"], axis=1)
#data = data.loc[(data['PERSON_AGE'] < 140) & (data['PERSON_AGE'] >= 0)]

with open("charts/description.txt", "w", encoding = 'utf-8') as f:
    f.write(str(data.describe()))

plt.figure(figsize=(10, 5))
data.boxplot(rot=45)
plt.tight_layout()
savefig(currentPath + "/images/global_boxplot.png", dpi=300)
show()

print(data.isna().sum() / (len(data)) * 100)  # Relative missing values
print("--")
print(data.isna().sum())  # Absolute missing values

# Printing box plots for each numeric variable
numeric_data = data.select_dtypes(include=np.number)
print(numeric_data.columns.tolist())

rows, cols = ds.choose_grid(len(numeric_data.columns.tolist()))
print("rows: {}, cols: {}".format(rows, cols))
fig, axs = plt.subplots(rows, cols, figsize=(rows * ds.HEIGHT, cols * ds.HEIGHT))
i, j = 0, 0


for n in range(len(numeric_data.columns.tolist())):
    print("for: {}".format(numeric_data.columns.tolist()[n]))

    axs[i,j].set_title('Boxplot for %s' % numeric_data.columns.tolist()[n])
    boxprops = dict(linestyle='-', linewidth=1, color='#005493')
    axs[i,j].boxplot(data[numeric_data.columns.tolist()[n]].dropna().values, boxprops=boxprops)

    i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
plt.savefig(currentPath + '/images/single_boxplots.png')
plt.show()
