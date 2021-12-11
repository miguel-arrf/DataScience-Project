import matplotlib.pyplot
import pandas
from matplotlib.pyplot import subplots, savefig, show
from pandas import read_csv

from ds_charts import choose_grid, HEIGHT

filename = '../data/NYC_collisions_tabular.csv'
data = read_csv(filename)

values = {'nr records': data.shape[0], 'nr variables': data.shape[1]}
safety_dict = {"NOT EQUIPPED": 0, "UPPER BODY": 1, "LOWER BODY": 2, "HEAD": 3, "AIRBAG": 4, "CHILD RESTRAINT": 5, "OTHER": 6, "HEAD & OTHER": 7, "UPPER BODY & AIRBAG": 8, "AIRBAG & CHILD RESTRAINT": 9}
injury_dict = {"NONE": 0, "LOWER BODY": 1, "UPPER BODY": 2, "HEAD": 3, "ENTIRE BODY": 4}

def transform_equipment(v):
    aux = ""
    if "nan" in str(v) or "None" in str(v) or "Unknown" in str(v):
        aux = "NOT EQUIPPED"
    else:
        if "Helmet" in str(v):
            aux += "HEAD & "
        if "Belt" in str(v) or "Harness" in str(v):
            aux += "UPPER BODY & "
        if "Pads" in str(v) or "Stoppers" in str(v):
            aux += "LOWER BODY & "
        if "Air Bag" in str(v):
            aux += "AIRBAG & "
        if "Child Restraint" in str(v):
            aux += "CHILD RESTRAINT & "
        if "Other" in str(v):
            aux += "OTHER"
    if aux[-3:] == " & ":
        aux = aux[:-3]
    v = aux
    return v

def transform_injury(v):
    aux = ""
    if "Does Not Apply" in str(v) or "Unknown" in str(v):
        aux = "NONE"
    if "Leg" in str(v):
        aux = "LOWER BODY"
    if "Eye" in str(v) or "Head" in str(v) or "Face" in str(v) or "Neck" in str(v):
        aux = "HEAD"
    if "Arm" in str(v) or "Back" in str(v) or "Chest" in str(v) or "Abdomen" in str(v):
        aux = "UPPER BODY"
    if "Entire Body" in str(v):
        aux = "ENTIRE BODY"
    v = aux
    return v


data['SAFETY_EQUIPMENT'] = data['SAFETY_EQUIPMENT'].apply(lambda x: transform_equipment(x))
data['BODILY_INJURY'] = data['BODILY_INJURY'].apply(lambda x: transform_injury(x))

injury_order = ["NONE", "LOWER BODY", "UPPER BODY", "HEAD", "ENTIRE BODY"]
equipment_order = ["NOT EQUIPPED", "UPPER BODY", "LOWER BODY", "HEAD", "AIRBAG", "CHILD RESTRAINT", "OTHER", "HEAD & OTHER", "UPPER BODY & AIRBAG", "AIRBAG & CHILD RESTRAINT"]

c_classes = pandas.api.types.CategoricalDtype(ordered = True, categories = equipment_order)
data['SAFETY_EQUIPMENT'] = data['SAFETY_EQUIPMENT'].astype(c_classes)
to_plot = data.SAFETY_EQUIPMENT.value_counts(sort=False)

c_classes2 = pandas.api.types.CategoricalDtype(ordered = True, categories = injury_order)
data['BODILY_INJURY'] = data['BODILY_INJURY'].astype(c_classes2)
to_plot2 = data.BODILY_INJURY.value_counts(sort=False)

variables = [to_plot, to_plot2]
variables2 = ['SAFETY_EQUIPMENT', 'BODILY_INJURY']

rows, cols = choose_grid(len(variables))
fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
i, j = 0, 0

matplotlib.pyplot.setp(axs)
matplotlib.pyplot.subplots_adjust(bottom=0.36)

for n in range(len(variables)):
    axs[i, j].set_title(f'Histogram for {variables2[n]}')
    axs[i, j].set_xlabel(variables2[n])
    axs[i, j].set_ylabel('nr records')
    axs[i, j].bar(variables[n].index, variables[n].values)
    for tick in axs[i, j].get_xticklabels():
        tick.set_rotation(90)
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)

savefig('../DataGranularity_Set1/images/taxonomy_single.png')
show()
