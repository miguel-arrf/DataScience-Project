import numpy as np
import seaborn
from pandas import read_csv
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import subplots, savefig
from ds_charts import get_variable_types, HEIGHT

register_matplotlib_converters()
filename = '../encodedWithGranularity.csv'
data = read_csv(filename, index_col='UNIQUE_ID', parse_dates=True, infer_datetime_format=True)
'''
category_variables_to_use = ["BODILY_INJURY", "SAFETY_EQUIPMENT", "PERSON_SEX", "PERSON_TYPE", "PED_LOCATION",
                             "CONTRIBUTING_FACTOR_2", "EJECTION", "COMPLAINT", "EMOTIONAL_STATUS",
                             "CONTRIBUTING_FACTOR_1",
                             "POSITION_IN_VEHICLE", "PED_ROLE", "PED_ACTION", "PERSON_INJURY"]
'''

symbolic_vars = get_variable_types(data)['Symbolic']
print("Symbolic vars: {}, tamanho: {}".format(symbolic_vars, len(symbolic_vars)))
symbolic_vars.remove("PERSON_ID")
symbolic_vars.remove("CRASH_TIME")
symbolic_vars.remove("CRASH_DATE")
symbolic_vars.append("PERSON_INJURY")

print("symbolic: ", symbolic_vars)

# symbolic_vars = ["BODILY_INJURY", 'SAFETY_EQUIPMENT', 'PERSON_SEX', 'PERSON_TYPE']

if not symbolic_vars:
    raise ValueError('There are no symbolic variables.')

colors = np.where(data["PERSON_INJURY"] == "Injured", 'goldenrod', 'red')

rows, cols = len(symbolic_vars) - 1, len(symbolic_vars) - 1
fig, axs = subplots(rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False)
for i in range(len(symbolic_vars)):
    var1 = symbolic_vars[i]
    print("var1: ", var1)
    for j in range(i + 1, len(symbolic_vars)):
        var2 = symbolic_vars[j]
        print("var2: ", var2)
        axs[i, j - 1].set_title("%s x %s" % (var1, var2))
        axs[i, j - 1].set_xlabel(var1)
        axs[i, j - 1].set_ylabel(var2)
        axs[i, j - 1].scatter(data[var1].astype(str), data[var2].astype(str), c=colors, cmap='winter')
print("All coiso")
savefig(f'../DataSparsity_Set1/images/sparsity_study_symbolic.png')
print("Done.")
