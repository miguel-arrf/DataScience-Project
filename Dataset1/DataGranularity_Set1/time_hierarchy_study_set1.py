from pandas import read_csv
from ds_charts import get_variable_types, choose_grid, HEIGHT
from matplotlib.pyplot import subplots, savefig, show
import datetime
from pandas.tseries.holiday import USFederalHolidayCalendar
import holidays

filename =   '../../data/NYC_collisions_tabular.csv'
data = read_csv(filename)
data = data.drop(["VEHICLE_ID", "COLLISION_ID", "PERSON_ID", "UNIQUE_ID"], axis=1)
data = data.loc[(data['PERSON_AGE'] < 140) & (data['PERSON_AGE'] >= 0)]

values = {'nr records': data.shape[0], 'nr variables': data.shape[1]}
date_dict = {"WEEKDAY": 0, "WEEKEND": 1, "HOLIDAY": 2}
time_dict = {"DAWN": 0, "MORNING": 1, "LUNCH TIME": 2, "AFTERNOON": 3, "DINNER TIME": 4, "NIGHT": 5}

cal = USFederalHolidayCalendar()
holidays = cal.holidays(start='2021-01-01', end='2021-11-17').to_pydatetime()


def transform_date(v):
    date = str(v).split("/")
    date_for_checking = datetime.datetime(day=int(date[0]), month=int(date[1]), year=int(date[2]))
    if date_for_checking.weekday() < 5:
        aux = "WEEKDAY"
    else:
        aux = "WEEKEND"
    if date_for_checking in holidays:
        aux = "HOLIDAY"
    v = date_dict[aux]
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
    v = time_dict[aux]
    return v


data['CRASH_DATE'] = data['CRASH_DATE'].apply(lambda x: transform_date(x))
data['CRASH_TIME'] = data['CRASH_TIME'].apply(lambda x: transform_time(x))
variables = ['CRASH_DATE', 'CRASH_TIME']

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
savefig('../DataGranularity_Set1/images/time_hierarchy_study.png')
show()
