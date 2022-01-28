import pandas

df = pandas.read_csv("air_quality_tabular.csv")
print(df['City_EN'].value_counts().idxmin())
print(df['City_EN'].value_counts().min())
print(df['City_EN'].value_counts().idxmax())
print(df['City_EN'].value_counts().max())
print(len(df['City_EN'].unique()))
df.drop(["FID"], axis=1, inplace=True)
#self.dataset.drop(["date"], axis=1, inplace=True)
df.drop(["City_EN"], axis=1, inplace=True)
df.drop(["Prov_EN"], axis=1, inplace=True)
#self.dataset.drop(["GbCity"], axis=1, inplace=True)
#self.dataset.drop(["GbProv"], axis=1, inplace=True)
df.drop(["Field_1"], axis=1, inplace=True)
print(df["GbCity"])
df["GbCity"].replace({"s": 3412}, inplace=True)
df = df.astype({"GbCity" : 'int64'})
print(df["GbCity"])
df = df.dropna()
print(df.shape)
