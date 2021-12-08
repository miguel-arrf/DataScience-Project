import pandas
from pandas import read_csv
from ds_charts import get_variable_types, choose_grid, HEIGHT
from matplotlib.pyplot import subplots, savefig, show
import matplotlib
import datetime
from pandas.tseries.holiday import USFederalHolidayCalendar

filename = '../data/NYC_collisions_tabular.csv'
data = read_csv(filename)

values = {'nr records': data.shape[0], 'nr variables': data.shape[1]}
date_dict = {"WEEKDAY": 0, "WEEKEND": 1, "HOLIDAY": 2}
time_dict = {"DAWN": 0, "MORNING": 1, "LUNCH TIME": 2, "AFTERNOON": 3, "DINNER TIME": 4, "NIGHT": 5}

def transform_date(v):
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start='2021-01-01', end='2021-11-17').to_pydatetime()
    date = str(v).split("/")
    date_for_checking = datetime.datetime(day=int(date[0]), month=int(date[1]), year=int(date[2]))
    if date_for_checking.weekday() < 5:
        aux = "WEEKDAY"
    else:
        aux = "WEEKEND"
    if date_for_checking in holidays:
        aux = "HOLIDAY"
    v = aux
    return v

def transform_time(v):
    aux = ""
    time = int(str(v).split(":")[0])
    if 0 <= time <= 6:
        aux = "DAWN"
    if 7 <= time <= 11:
        aux = "MORNING"
    if 12 <= time <= 13:
        aux = "LUNCH TIME"
    if 14 <= time <= 18:
        aux = "AFTERNOON"
    if 19 <= time <= 20:
        aux = "DINNER TIME"
    if 21 <= time <= 23:
        aux = "NIGHT"
    v = aux
    return v


data['CRASH_DATE'] = data['CRASH_DATE'].apply(lambda x: transform_date(x))
data['CRASH_TIME'] = data['CRASH_TIME'].apply(lambda x: transform_time(x))

date_order = ["WEEKDAY", "WEEKEND", "HOLIDAY"]
time_order = ["DAWN", "MORNING", "LUNCH TIME", "AFTERNOON", "DINNER TIME", "NIGHT"]

c_classes = pandas.api.types.CategoricalDtype(ordered = True, categories = date_order)
data['CRASH_DATE'] = data['CRASH_DATE'].astype(c_classes)
to_plot = data.CRASH_DATE.value_counts(sort=False)

c_classes2 = pandas.api.types.CategoricalDtype(ordered = True, categories = time_order)
data['CRASH_TIME'] = data['CRASH_TIME'].astype(c_classes2)
to_plot2 = data.CRASH_TIME.value_counts(sort=False)

variables = [to_plot, to_plot2]
variables2 = ['CRASH_DATE', 'CRASH_TIME']

rows, cols = choose_grid(len(variables))
fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
i, j = 0, 0
matplotlib.pyplot.setp(axs)
matplotlib.pyplot.subplots_adjust(bottom=0.25)

for n in range(len(variables)):
    axs[i, j].set_title(f'Histogram for {variables2[n]}')
    axs[i, j].set_xlabel(variables2[n])
    axs[i, j].set_ylabel('nr records')
    axs[i, j].bar(variables[n].index, variables[n].values)
    for tick in axs[i, j].get_xticklabels():
        tick.set_rotation(90)
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)

savefig('../DataGranularity_Set1/images/time_hierarchy_single.png')
show()

