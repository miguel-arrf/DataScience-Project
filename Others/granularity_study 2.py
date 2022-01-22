import pandas
from pandas import read_csv, DataFrame

filename = '../data/NYC_collisions_tabular.csv'
df = read_csv(filename, index_col='UNIQUE_ID', na_values='', parse_dates=True,
                   infer_datetime_format=True)
columns = {}
for col in df:
    if "ID" not in col and "CRASH_DATE" not in col and "CRASH_TIME" not in col and "PERSON_AGE" not in col:
        print("Col: ", col)
        print(df[col].unique())
        print()
        columns[col] = df[col].unique()

new_df = DataFrame()

for col in columns:
    new_df = pandas.concat([new_df, DataFrame({col:columns[col]})], axis=1)

new_df.to_csv('teste.csv')
