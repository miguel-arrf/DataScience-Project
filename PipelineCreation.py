import pandas
from pandas import read_csv
from sklearn.preprocessing import OneHotEncoder

from DataGranularity_Set1 import taxonomy_function_apply_set1
from DataGranularity_Set1 import timeHierarchy_function_apply_set1
from MissingValuesImputation_Set1 import imputation_function_apply_set1
from Dummification_Set1 import dummification_function_apply_set1
import category_encoders as ce


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

    def save_pipeline(self):
        self.dataset.to_csv("data/Set1_noAnomalies_TimeHierarchy.csv")

    def missing_values_imputation(self):
        imputation_function_apply_set1.apply_set1_imputation(self.dataset)

    def dummification(self):
        df_one = pandas.get_dummies(self.dataset["PERSON_INJURY"])
        self.dataset["PERSON_INJURY"] = df_one["Injured"]
        person_injury = self.dataset["PERSON_INJURY"]
        self.dataset = dummification_function_apply_set1.apply_set1_imputation(self.dataset,
                                                                               columns=["BODILY_INJURY",
                                                                                        "SAFETY_EQUIPMENT"],
                                                                               type_of_encoder=ce.TargetEncoder)
        print("<---------->")
        self.dataset["PERSON_INJURY"] = person_injury
        print("<--->")
        for column in self.dataset.columns:
            print(column)


if __name__ == '__main__':
    pipeline = CD_Pipeline_Set1()
    print(pipeline.get_original_dataset().shape)
    pipeline.remove_incorrect_values()
    # pipeline.set_safety_equipement_granularity()
    pipeline.set_bodily_injury_granularity()
    pipeline.missing_values_imputation()
    pipeline.dummification()
    pipeline.save_pipeline()
    print(pipeline.get_original_dataset().shape)
