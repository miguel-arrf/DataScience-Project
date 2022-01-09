import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from pandas import DataFrame, concat
from pandas import read_csv
from sklearn.preprocessing import *

from ds_charts import get_variable_types

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


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
    return final_df


class Pipeline:
    def __init__(self):
        filename = '../data/NYC_collisions_tabular.csv'
        self.dataset = read_csv(filename, index_col='UNIQUE_ID', na_values='', parse_dates=True,
                                infer_datetime_format=True)

        print(self.dataset['EMOTIONAL_STATUS'].unique())

        self.dataset.drop(["COLLISION_ID"], axis=1, inplace=True)

        self.bodilyInjuryEncoder = None
        self.safetyEquipmentEncoder = None

        self.personTypeEncoder = None
        self.ejectionEncoder = None

        self.complaintEncoder = None
        self.emotionalStatusEncoder = None
        self.pedRoleEncoder = None
        self.pedActionEncoder = None

        self.pedLocationEncoder = None

    def removeIncorrectValues(self):
        self.dataset.drop(self.dataset[(self.dataset.PERSON_AGE < 0) | (self.dataset.PERSON_AGE > 200)].index,
                          inplace=True)

    def groupTime(self):
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

        self.dataset['CRASH_TIME'] = self.dataset['CRASH_TIME'].apply(
            lambda x: transform_time(x))

    def groupSafetyEquipment(self):
        def transform_safetyEquipment(v):
            aux = None
            if "Air Bag" in v:
                aux = "AirBag"
            elif "Lap Belt" in v:
                aux = "Belt"
            elif "Helmet" in v:
                aux = "Helmet"
            if aux is None:
                return v
            else:
                return aux

        self.dataset['SAFETY_EQUIPMENT'] = self.dataset['SAFETY_EQUIPMENT'].apply(
            lambda x: transform_safetyEquipment(x))

    def save_pipeline(self, datasetName):
        self.dataset.to_csv("data/{}.csv".format(datasetName))
        print("saved: {}, with shape:  {}".format(datasetName, self.dataset.shape))

    def saveTrainAndTestData(self, testName, trainName):
        df = self.dataset.copy()

        symbolic_vars = get_variable_types(df)['Symbolic']
        for symbolic_var in symbolic_vars:
            df[symbolic_var] = pd.factorize(df[symbolic_var])[0]

        df["PERSON_SEX"].replace(('F', 'M'), (1, 0), inplace=True)

        killed = df.loc[df["PERSON_INJURY"] == 'Killed']
        injured = df.loc[df["PERSON_INJURY"] == 'Injured']

        trainKilled, validate, test = np.split(killed.sample(frac=1, random_state=42),
                                               [int(.7 * len(killed)), int(.9 * len(killed))])
        testKilled = pd.concat([validate, test])

        trainInjured, validate, test = np.split(injured.sample(frac=1, random_state=42),
                                                [int(.7 * len(injured)), int(.9 * len(injured))])
        testInjured = pd.concat([validate, test])

        train = pd.concat([trainKilled, trainInjured])
        test = pd.concat([testKilled, testInjured])

        print("Train: ", train.shape)
        print("Test: ", test.shape)

        test.to_csv("data/{}.csv".format(testName))
        self.smote(train).to_csv("data/{}.csv".format(trainName))

    def smote(self, train):
        RANDOM_STATE = 42
        class_var = 'PERSON_INJURY'

        smote = SMOTE(sampling_strategy='minority', random_state=RANDOM_STATE, n_jobs=-1)
        y = train.pop(class_var).values
        X = train.values

        smote_X, smote_y = smote.fit_resample(X, y)
        df_smote = pd.concat([DataFrame(smote_X), DataFrame(smote_y)], axis=1)
        df_smote.columns = list(train.columns) + [class_var]
        return df_smote


if __name__ == '__main__':
    pipeline = Pipeline()
    print("Initial pipeline shape: ", pipeline.dataset.shape)

    pipeline.removeIncorrectValues()

    pipeline.groupSafetyEquipment()
    pipeline.groupTime()



    pipeline.saveTrainAndTestData("test", "train")
