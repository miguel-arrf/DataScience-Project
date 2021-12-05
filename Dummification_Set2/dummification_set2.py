from pandas import read_csv, DataFrame, concat
from pandas.plotting import register_matplotlib_converters
from ds_charts import get_variable_types
from sklearn.preprocessing import OneHotEncoder
from numpy import number

register_matplotlib_converters()
file = 'air_quality_tabular'
filename = '../data/air_quality_tabular.csv'
data = read_csv(filename, index_col='UNIQUE_ID', na_values='', parse_dates=True, infer_datetime_format=True)

# Drop out all records with missing values
data.dropna(inplace=False)


def dummify(df, vars_to_dummify):
    other_vars = [c for c in df.columns if not c in vars_to_dummify]
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False, dtype=bool)
    X = df[vars_to_dummify]
    encoder.fit(X)
    new_vars = encoder.get_feature_names(vars_to_dummify)
    trans_X = encoder.transform(X)
    dummy = DataFrame(trans_X, columns=new_vars, index=X.index)
    dummy = dummy.convert_dtypes(convert_boolean=True)
    final_df = concat([df[other_vars], dummy], axis=1)
    print(final_df.shape)
    return final_df


variables = get_variable_types(data)
symbolic_vars = ["City_EN", "Prov_EN"]
df = dummify(data, symbolic_vars)
# df.to_csv(f'../Dummification_Set2/data/{file}_dummified.csv', index=False)

df.describe(include=[bool])

# NOTE: The dataframe is too big to be uploaded to github...
# TODO: On this tutorial, there's something about class variables and that we shouldn't mess with them... Are they
#  the target variables?
