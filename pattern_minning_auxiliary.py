import pandas
import pandas as pd
from pandas import concat, DataFrame
from sklearn.preprocessing import OneHotEncoder

from ds_charts import get_variable_types


def dummify(df, vars_to_dummify):
    df = df.copy()
    other_vars = [c for c in df.columns if not c in vars_to_dummify]
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False, dtype=bool)
    X = df[vars_to_dummify]
    for column in X.columns:
        X[column] = X[column].apply(str)
        X[column] = X[column].astype("string")
    encoder.fit(X)
    new_vars = encoder.get_feature_names(vars_to_dummify)
    trans_X = encoder.transform(X)
    dummy = DataFrame(trans_X, columns=new_vars, index=X.index)
    dummy = dummy.convert_dtypes(convert_boolean=True)

    final_df = concat([df[other_vars], dummy], axis=1)
    return final_df

def dummifyDataset(data):
    dataset = data.copy()
    variables = get_variable_types(dataset)['Symbolic']
    print(variables)



    df = dummify(dataset,variables)
    return df

'''
set1_equal_width = pandas.read_csv("/Users/miguelferreira/Downloads/novosFicheiros/set1_equal_width.csv")
set1_equal_frequency = pandas.read_csv("/Users/miguelferreira/Downloads/novosFicheiros/set1_equal_freq.csv")


d1_1 = dummifyDataset(set1_equal_width)
print(d1_1.shape)
d1_1.to_csv("set1_equal_width_dummy.csv")
print("Done")

d1_2 = dummifyDataset(set1_equal_frequency)
print(d1_2.shape)
d1_2.to_csv("set1_equal_frequency_dummy.csv")
print("Done")

set2_equal_width = pandas.read_csv("/Users/miguelferreira/Downloads/novosFicheiros/set2_equal_width.csv")
set2_equal_fequency = pandas.read_csv("/Users/miguelferreira/Downloads/novosFicheiros/set2_equal_freq.csv")

d2_1 = dummifyDataset(set2_equal_width)
print(d2_1.shape)
d2_1.to_csv("set2_equal_width_dummy.csv")
print("Done")

d2_2 = dummifyDataset(set2_equal_fequency)
print(d2_2.shape)
d2_2.to_csv("set2_equal_frequency_dummy.csv")
print("Done")


print(pandas.read_csv("set2_equal_frequency_dummy.csv").shape)
print(pandas.read_csv("set2_equal_width_dummy.csv").shape)
'''

set2_default = pandas.read_csv("/Users/miguelferreira/Downloads/novosFicheiros/set2_default.csv")
set2_default.pop("date")
print(set2_default.shape)
d2_3 = dummifyDataset(set2_default)
d2_3.to_csv("set2_default_dummy.csv")
print(pandas.read_csv("set2_default_dummy.csv").shape)

