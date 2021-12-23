import pandas as pd

#filename = 'data/NYC_collisions_tabular.csv'
#data = pd.read_csv(filename, index_col='UNIQUE_ID', na_values='', parse_dates=True, infer_datetime_format=True)
#a = data['SAFETY_EQUIPMENT'].unique()



import datetime

from chinese_holiday import is_holiday, get_holiday_data
from pandas import read_csv

'''
def sunday_saturday_definer():
    date_for_checking = datetime.datetime(day = 30, month= 12, year= 2021)
    print(date_for_checking.weekday())

sunday_saturday_definer()


#print(is_holiday('2021-06-14'))


def is_chinese_holiday(v):
    date = str(v).split("/")
    date_for_checking = datetime.datetime(day=int(date[2]), month=int(date[1]), year=int(date[0]))
    if(is_holiday(date_for_checking)):
        return True
    else:
        return False

print(is_chinese_holiday('2021/11/14'))


def is_holiday_2021(date):
    holidays_to_return = []
    holidays = get_holiday_data('2020') + get_holiday_data('2021')
    for holiday in holidays:
        if not holiday[2]:
            holidays_to_return.append((holiday[0], holiday[1]))
    return holidays_to_return


print(is_holiday_2021(''))
'''
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

filename = 'data/NYC_collisions_tabular.csv'
df = read_csv(filename, index_col='UNIQUE_ID', na_values='', parse_dates=True,infer_datetime_format=True)
#print(df[df.duplicated(['VEHICLE_ID'], keep=False)]["VEHICLE_ID"])

print(df[df.duplicated(['VEHICLE_ID'], keep=False)])
