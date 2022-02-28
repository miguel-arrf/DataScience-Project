import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import subplots, savefig, show
from ds_charts import get_variable_types, HEIGHT

register_matplotlib_converters()
filename = '../../data/air_quality_tabular_granularity.csv'
data = read_csv(filename)
data.drop(["FID"], axis=1, inplace=True)
# self.dataset.drop(["date"], axis=1, inplace=True)
data.drop(["City_EN"], axis=1, inplace=True)
data.drop(["Prov_EN"], axis=1, inplace=True)
# self.dataset.drop(["GbCity"], axis=1, inplace=True)
# self.dataset.drop(["GbProv"], axis=1, inplace=True)
data.drop(["Field_1"], axis=1, inplace=True)
data["GbCity"].replace({"s": 3412}, inplace=True)
data = data.astype({"GbCity": 'int64'})
numeric_vars = get_variable_types(data)['Numeric']
numeric_vars.append('ALARM')
if [] == numeric_vars:
    raise ValueError('There are no numeric variables.')

rows, cols = len(numeric_vars) - 1, len(numeric_vars) - 1
fig, axs = subplots(rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False)

colors = np.where(data["ALARM"] == "Safe", 'g', 'r')

for i in range(len(numeric_vars)):
    var1 = numeric_vars[i]
    print("var1: ", var1)
    for j in range(i + 1, len(numeric_vars)):
        var2 = numeric_vars[j]
        axs[i, j - 1].set_title("%s x %s" % (var1, var2))
        axs[i, j - 1].set_xlabel(var1)
        axs[i, j - 1].set_ylabel(var2)
        axs[i, j - 1].scatter(data[var1], data[var2], c=colors, cmap='winter')
print("Saving figure")
plt.tight_layout()
savefig(f'../DataSparsity_Set2/images/sparsity_study_numeric.png')
# show()
print("Done")
