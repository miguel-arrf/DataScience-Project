import matplotlib.pyplot as plt
from pandas import read_csv
from ds_charts import get_variable_types, choose_grid, HEIGHT
from matplotlib.pyplot import subplots, savefig, show
import datetime
from chinese_holiday import get_holiday_data

filename = '../data/air_quality_tabular.csv'
data = read_csv(filename)

values = {'nr records': data.shape[0], 'nr variables': data.shape[1]}
date_dict = {"WEEKDAY": 0, "WEEKEND": 1, "HOLIDAY": 2}


def is_holiday(date):
    holidays_list = get_holidays2020_2021()
    aux = False
    for holiday in holidays_list:
        min_date_aux = holiday[0].split("-")
        max_date_aux = holiday[1].split("-")
        min_date = datetime.datetime(day=int(min_date_aux[2]), month=int(min_date_aux[1]), year=int(min_date_aux[0]))
        max_date = datetime.datetime(day=int(max_date_aux[2]), month=int(max_date_aux[1]), year=int(max_date_aux[0]))
        if min_date <= date <= max_date:
            aux = True
    return aux

def get_holidays2020_2021():
    holidays_to_return = []
    holidays = get_holiday_data('2020') + get_holiday_data('2021')
    for holiday in holidays:
        if not holiday[2]:
            holidays_to_return.append((holiday[0], holiday[1]))
    return holidays_to_return

def transform_date(v):
    date = str(v).split("/")
    date_for_checking = datetime.datetime(day=int(date[0]), month=int(date[1]), year=int(date[2]))
    if date_for_checking.weekday() < 5:
        aux = "WEEKDAY"
    else:
        aux = "WEEKEND"
    if is_holiday(date_for_checking):
        aux = "HOLIDAY"
    v = date_dict[aux]
    return v

data['date'] = data['date'].apply(lambda x: transform_date(x))
variables = ['date']

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
plt.tight_layout()
savefig('../DataGranularity_Set2/images/time_hierarchy_study.png', dpi=300)
show()
