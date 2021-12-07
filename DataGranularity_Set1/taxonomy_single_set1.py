from pandas import read_csv
from ds_charts import get_variable_types, choose_grid, HEIGHT
from matplotlib.pyplot import subplots, savefig, show

filename = '../data/NYC_collisions_tabular.csv'
data = read_csv(filename)

values = {'nr records': data.shape[0], 'nr variables': data.shape[1]}
safety_dict = {'NOT EQUIPPED': 0, "UPPER BODY": 1, "LOWER BODY": 2, "HEAD": 3, "AIRBAG": 4, "CHILD RESTRAINT": 5, "OTHER": 6, "HEAD & OTHER": 7, "UPPER BODY & AIRBAG": 8, "AIRBAG & CHILD RESTRAINT": 9}
def transform(v):
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

data['SAFETY_EQUIPMENT'] = data['SAFETY_EQUIPMENT'].apply(lambda x: transform(x))
variables = ['SAFETY_EQUIPMENT']

rows, cols = choose_grid(len(variables))
fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
i, j = 0, 0

for n in range(len(variables)):
    axs[i, j].set_title(f'Histogram for {variables[n]}')
    axs[i, j].set_xlabel(variables[n])
    axs[i, j].set_ylabel('nr records')
    axs[i, j].hist(data[variables[n]].astype(int).values, bins=100)
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
savefig('../DataGranularity_Set1/images/taxonomy_single.png')
show()
