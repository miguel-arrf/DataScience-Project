import category_encoders as ce
import pandas
from matplotlib.pyplot import figure, show
from pandas import read_csv
from pandas import DataFrame, concat


from sklearn.preprocessing import *
from DataGranularity_Set1 import taxonomy_function_apply_set1
from DataGranularity_Set1 import timeHierarchy_function_apply_set1
from Dummification_Set1 import dummification_function_apply_set1
from MissingValuesImputation_Set1 import imputation_function_apply_set1
from ds_charts import get_variable_types, multiple_bar_chart, HEIGHT
from Balancing_Set1 import pipeline_balancing

class CD_Pipeline_Set1:
    def __init__(self):
        filename = 'data/NYC_collisions_tabular.csv'
        self.dataset = read_csv(filename, index_col='UNIQUE_ID', na_values='', parse_dates=True,
                                infer_datetime_format=True)

    def get_original_dataset(self):
        return self.dataset

    def remove_incorrect_values(self):
        self.dataset.drop(self.dataset[(self.dataset.PERSON_AGE < 0) | (self.dataset.PERSON_AGE > 200)].index,
                          inplace=True)

    def set_safety_equipement_granularity(self):
        taxonomy_set1 = taxonomy_function_apply_set1.taxonomy_set1()
        taxonomy_set1.apply_set1_taxonomy(self.dataset)

    def set_bodily_injury_granularity(self):
        timeHierarchy_function_apply_set1.apply_set1_taxonomy(self.dataset)

    def save_pipeline(self, datasets):
        if datasets is not None:
            for key, dataset in datasets.items():
                dataset[0].to_csv("data/set1_pipeline/{}.csv".format(key
                                                                     .replace(" ", "")
                                                                     .replace("<", "")
                                                                     .replace("'", "")
                                                                     .replace(">", "")
                                                                     .replace("category_encoders", "")
                                                                     .replace("class,", "")
                                                                     ))
                print("saved: {}, with shape:  {}".format(key, dataset[0].shape))

        else:
            self.dataset.to_csv("data/Set1_noAnomalies_TimeHierarchy.csv")

    def missing_values_imputation(self):
        # Vechicle_ID ->
        imputation_function_apply_set1.apply_set1_imputation(self.dataset)
        self.dataset = self.dataset[self.dataset['PERSON_AGE'].notna()]
        self.dataset.drop(self.dataset.loc[self.dataset['PERSON_SEX'] == "U"].index, inplace=True)

    def calculate_outliers(self, datasets):
        to_return = {}
        for key, tuple in datasets.items():
            dataset = tuple[0]
            NR_STDEV: int = 2
            numeric_vars = get_variable_types(dataset)['Numeric']

            outliers_iqr = []
            outliers_stdev = []
            summary5 = dataset.describe(include='number')

            for var in numeric_vars:
                iqr = 1.5 * (summary5[var]['75%'] - summary5[var]['25%'])
                outliers_iqr += [
                    dataset[dataset[var] > summary5[var]['75%'] + iqr].count()[var] +
                    dataset[dataset[var] < summary5[var]['25%'] - iqr].count()[var]]
                std = NR_STDEV * summary5[var]['std']
                outliers_stdev += [
                    dataset[dataset[var] > summary5[var]['mean'] + std].count()[var] +
                    dataset[dataset[var] < summary5[var]['mean'] - std].count()[var]]

                summary6 = dataset.describe(include='number')
                if var == "PERSON_AGE":
                    outliers_to_get_1 = dataset[dataset[var] > summary6[var]['75%'] + iqr]
                    outliers_to_get_2 = dataset[dataset[var] < summary6[var]['25%'] - iqr]

                    ages_to_remove = []
                    for age in outliers_to_get_1["PERSON_AGE"].unique():
                        ages_to_remove.append(age)
                    for age in outliers_to_get_2["PERSON_AGE"].unique():
                        ages_to_remove.append(age)

                    #print("ages to remove: ", ages_to_remove)

                    dataset = dataset[~dataset["PERSON_AGE"].isin(ages_to_remove)]
                    #print("new shape: ", dataset.shape)

            to_return[key] = (dataset, tuple[1])
            '''
            for i in range(len(numeric_vars)):
                if outliers_iqr[i] > outliers_stdev[i]:
                    print("Biggest for: {}, is: iqr".format(numeric_vars[i]))
                else:
                    print("Biggest for: {}, is: stdev".format(numeric_vars[i]))
            '''

            outliers = {'iqr': outliers_iqr, 'stdev': outliers_stdev}
            figure(figsize=(16, HEIGHT))
            multiple_bar_chart(numeric_vars, outliers, title='Nr of outliers per variable for {}'.format(key),
                               xlabel='variables',
                               ylabel='nr outliers', percentage=False)
            show()
        return to_return

    def dummification(self, encoders):
        df_one = pandas.get_dummies(self.dataset["PERSON_INJURY"])
        df_two = pandas.get_dummies(self.dataset["PERSON_SEX"])

        # boolean_vars = ["PERSON_INJURY", "PERSON_SEX"]
        print(df_two.columns)
        print(df_one.columns)
        self.dataset["PERSON_INJURY"] = df_one["Killed"]
        self.dataset["PERSON_SEX"] = df_two["M"]

        person_injury = self.dataset["PERSON_INJURY"]
        person_sex = self.dataset["PERSON_SEX"]

        symbolic_vars = ["BODILY_INJURY", "SAFETY_EQUIPMENT", "PERSON_TYPE", "PED_LOCATION", "CONTRIBUTING_FACTOR_2",
                         "EJECTION", "COMPLAINT", "EMOTIONAL_STATUS", "CONTRIBUTING_FACTOR_1", "POSITION_IN_VEHICLE",
                         "PED_ROLE", "PED_ACTION"]

        encoded_datasets = {}
        for encoder in encoders:
            temp_dataset = self.dataset.copy()
            temp_dataset = dummification_function_apply_set1.apply_set1_imputation(temp_dataset,
                                                                                   columns=symbolic_vars,
                                                                                   type_of_encoder=encoder)
            encoded_datasets["{}".format(encoder)] = temp_dataset
            temp_dataset[0]["PERSON_INJURY"] = person_injury
            temp_dataset[0]["PERSON_SEX"] = person_sex
            #print("number of columns: ", temp_dataset[0].shape)

        '''
        print("<---------->")
        self.dataset["PERSON_INJURY"] = person_injury
        self.dataset["PERSON_SEX"] = person_sex
        print("<--->")
        for column in self.dataset.columns:
            print(column)
        '''

        return encoded_datasets

    def scale_values(self, datasets):
        to_return = {}
        for key, tuple in datasets.items():
            dataset = tuple[0]
            variable_types = get_variable_types(dataset)
            numeric_vars = variable_types['Numeric']
            symbolic_vars = variable_types['Symbolic']
            boolean_vars = variable_types['Binary']

            df_nr = dataset[numeric_vars]
            df_sb = dataset[symbolic_vars]
            df_bool = dataset[boolean_vars]

            scalers = [StandardScaler(),  # Defaults = true for copy, with_mean and with_std
                       MinMaxScaler(feature_range=(0, 1)),
                       MaxAbsScaler(),
                       StandardScaler(with_mean=False),
                       RobustScaler()]


            for scaler in scalers:
                print("For {}, scaler: {}".format(key, scaler))
                scaler = scaler.fit(df_nr)
                tmp = DataFrame(scaler.transform(df_nr), index=dataset.index, columns=numeric_vars)
                norm_data_zscore = concat([tmp, df_sb, df_bool], axis=1)

                to_return["{}<->{}".format(key,scaler)] = (norm_data_zscore, tuple[1])

        return to_return




if __name__ == '__main__':
    pipeline = CD_Pipeline_Set1()
    print("Original shape: ", pipeline.get_original_dataset().shape)
    pipeline.remove_incorrect_values()
    # pipeline.set_safety_equipement_granularity()
    pipeline.set_bodily_injury_granularity()
    pipeline.missing_values_imputation()
    datasets = pipeline.dummification(encoders=[ce.OneHotEncoder,
                                                ce.LeaveOneOutEncoder,
                                                ce.BaseNEncoder])
    datasets = pipeline.calculate_outliers(datasets)
    datasets = pipeline.scale_values(datasets)
    pipeline.save_pipeline(datasets)
    print(pipeline.get_original_dataset().shape)
