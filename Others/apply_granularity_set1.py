import datetime

from pandas import read_csv
from pandas.tseries.holiday import USFederalHolidayCalendar

file = 'NYC_collisions_tabular'
filename = 'data/NYC_collisions_tabular.csv'
data = read_csv(filename)

values = {'nr records': data.shape[0], 'nr variables': data.shape[1]}
safety_dict = {"NOT EQUIPPED": 0, "UPPER BODY": 1, "LOWER BODY": 2, "HEAD": 3, "AIRBAG": 4, "CHILD RESTRAINT": 5, "OTHER": 6, "HEAD & OTHER": 7, "UPPER BODY & AIRBAG": 8, "AIRBAG & CHILD RESTRAINT": 9}
injury_dict = {"NONE": 0, "LOWER BODY": 1, "UPPER BODY": 2, "HEAD": 3, "ENTIRE BODY": 4}
date_dict = {"WEEKDAY": 0, "WEEKEND": 1, "HOLIDAY": 2}
time_dict = {"DAWN": 0, "MORNING": 1, "LUNCH TIME": 2, "AFTERNOON": 3, "DINNER TIME": 4, "NIGHT": 5}

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
data['SAFETY_EQUIPMENT'] = data['SAFETY_EQUIPMENT'].apply(lambda x: transform_equipment(x))
data['BODILY_INJURY'] = data['BODILY_INJURY'].apply(lambda x: transform_injury(x))

data.to_csv(f'data/{file}_granularity.csv', index=False)
