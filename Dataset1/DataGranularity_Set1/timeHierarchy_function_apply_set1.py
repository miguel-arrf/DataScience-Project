import datetime

import pandas
from pandas.tseries.holiday import USFederalHolidayCalendar


def apply_set1_taxonomy(data):
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

    c_classes = pandas.api.types.CategoricalDtype(ordered=True, categories=date_order)
    data['CRASH_DATE'] = data['CRASH_DATE'].astype(c_classes)

    c_classes2 = pandas.api.types.CategoricalDtype(ordered=True, categories=time_order)
    data['CRASH_TIME'] = data['CRASH_TIME'].astype(c_classes2)
