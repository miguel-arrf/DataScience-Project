from pandas import read_csv
from ds_charts import get_variable_types, choose_grid, HEIGHT
from matplotlib.pyplot import subplots, savefig, show

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
    v = safety_dict[aux]
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
    v = injury_dict[aux]
    return v


data['SAFETY_EQUIPMENT'] = data['SAFETY_EQUIPMENT'].apply(lambda x: transform_equipment(x))
data['BODILY_INJURY'] = data['BODILY_INJURY'].apply(lambda x: transform_injury(x))
variables = ['BODILY_INJURY', 'SAFETY_EQUIPMENT']

rows = len(variables)
bins = (2, 5, 10, 100, 1000)
cols = len(bins)
fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)

for i in range(rows):
    for j in range(cols):
        axs[i, j].set_title(f'Histogram for {variables[i]} {bins[j]} bins')
        axs[i, j].set_xlabel(variables[i])
        axs[i, j].set_ylabel('Nr records')
        axs[i, j].hist(data[variables[i]].values, bins=bins[j])
savefig('../DataGranularity_Set1/images/taxonomy_study.png')
show()