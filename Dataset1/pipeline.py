import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from pandas import DataFrame, concat
from pandas import read_csv
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import *

from MissingValuesImputation_Set1 import imputation_function_apply_set1
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

    def encode_PedAction(self):
        categories = self.dataset["PED_ACTION"].unique()
        sortedCategories = ["Crossing,No Signal,or Crosswalk", "Crossing With Signal", "Other Actions in Roadway"]
        for category in categories:
            if category not in sortedCategories:
                sortedCategories.append(category)

        sortedCategories = [np.array(list(reversed(sortedCategories)))]
        self.pedActionEncoder = OrdinalEncoder(categories=sortedCategories)
        self.dataset["PED_ACTION"] = self.pedActionEncoder.fit_transform(self.dataset[["PED_ACTION"]])

    def encode_PedRole(self):
        categories = self.dataset["PED_ROLE"].unique()
        sortedCategories = ["Pedestrian", "Passanger", "Driver"]
        for category in categories:
            if category not in sortedCategories:
                sortedCategories.append(category)

        sortedCategories = [np.array(list(reversed(sortedCategories)))]
        self.pedRoleEncoder = OrdinalEncoder(categories=sortedCategories)
        self.dataset["PED_ROLE"] = self.pedRoleEncoder.fit_transform(self.dataset[["PED_ROLE"]])

    def encode_EmotionalStatus(self):
        categories = self.dataset["EMOTIONAL_STATUS"].unique()
        sortedCategories = ["Apparent Death", "Unconscious", "Conscious"]
        for category in categories:
            if category not in sortedCategories:
                sortedCategories.append(category)

        sortedCategories = [np.array(list(reversed(sortedCategories)))]
        self.emotionalStatusEncoder = OrdinalEncoder(categories=sortedCategories)
        self.dataset["EMOTIONAL_STATUS"] = self.emotionalStatusEncoder.fit_transform(self.dataset[["EMOTIONAL_STATUS"]])

    def encode_Complaint(self):
        categories = self.dataset["COMPLAINT"].unique()
        sortedCategories = ["Internal", "Complaint of Pain or Nausea",
                            "None Visible",
                            "Crush Injuries",
                            "Contusion - Bruise",
                            "Severe Bleeding"]
        for category in categories:
            if category not in sortedCategories:
                sortedCategories.append(category)

        sortedCategories = [np.array(list(reversed(sortedCategories)))]
        self.complaintEncoder = OrdinalEncoder(categories=sortedCategories)
        self.dataset["COMPLAINT"] = self.complaintEncoder.fit_transform(self.dataset[["COMPLAINT"]])

    def encode_Ejection(self):
        categories = self.dataset["EJECTION"].unique()
        sortedCategories = ["Ejected", "Not Ejected"]
        for category in categories:
            if category not in sortedCategories:
                sortedCategories.append(category)

        sortedCategories = [np.array(list(reversed(sortedCategories)))]
        self.ejectionEncoder = OrdinalEncoder(categories=sortedCategories)
        self.dataset["EJECTION"] = self.ejectionEncoder.fit_transform(self.dataset[["EJECTION"]])

    def encode_PersonType(self):
        categories = self.dataset["PERSON_TYPE"].unique()
        sortedCategories = ["Occupant", "Pedestrian"]
        for category in categories:
            if category not in sortedCategories:
                sortedCategories.append(category)

        sortedCategories = [np.array(list(reversed(sortedCategories)))]
        self.personTypeEncoder = OrdinalEncoder(categories=sortedCategories)
        self.dataset["PERSON_TYPE"] = self.personTypeEncoder.fit_transform(self.dataset[["PERSON_TYPE"]])

    def removeIncorrectValues(self):
        self.dataset.drop(self.dataset[(self.dataset.PERSON_AGE < 0) | (self.dataset.PERSON_AGE > 200)].index,
                          inplace=True)

    def encode_BodilyInjury(self):
        categories = self.dataset["BODILY_INJURY"].unique()
        sortedCategories = ["Head", "Entire Body", "Back", "Knee-Lower Leg Foot", "Neck"]
        for category in categories:
            if category not in sortedCategories:
                sortedCategories.append(category)

        sortedCategories = [np.array(list(reversed(sortedCategories)))]

        self.bodilyInjuryEncoder = OrdinalEncoder(categories=sortedCategories)
        self.dataset["BODILY_INJURY"] = self.bodilyInjuryEncoder.fit_transform(self.dataset[["BODILY_INJURY"]])

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

    def encode_PedLocationAndPositionInVehicle(self):
        categories = self.dataset["PED_LOCATION"].unique()
        sortedCategories = ["Pedestrian/Bicyclist/Other Pedestrian Not at Intersection", "Pedestrian/Bicyclist/Other Pedestrian at Intersection", "Does Not Apply", "Unknown", "NotApplicable"]
        for category in categories:
            if category not in sortedCategories:
                sortedCategories.append(category)

        sortedCategories = [np.array(list(reversed(sortedCategories)))]
        print("sorted: ", sortedCategories)
        self.pedLocationEncoder = OrdinalEncoder(categories=sortedCategories)
        self.dataset["PED_LOCATION"] = self.pedLocationEncoder.fit_transform(self.dataset[["PED_LOCATION"]])

        #one_hot = pandas.get_dummies(self.dataset['PED_LOCATION'])
        #self.dataset = self.dataset.drop('PED_LOCATION', axis=1)
        #self.dataset = self.dataset.join(one_hot)


    def encode_SafetyEquipment(self):
        categories = self.dataset["SAFETY_EQUIPMENT"].unique()
        sortedCategories = ["None", "Helmet", "Unknown", "AirBag", "Belt"]
        for category in categories:
            if category not in sortedCategories:
                sortedCategories.append(category)

        sortedCategories = [np.array(list(reversed(sortedCategories)))]
        self.safetyEquipmentEncoder = OrdinalEncoder(categories=sortedCategories)
        self.dataset["SAFETY_EQUIPMENT"] = self.safetyEquipmentEncoder.fit_transform(self.dataset[["SAFETY_EQUIPMENT"]])

    def missing_values_imputation(self):
        # Vechicle_ID ->
        imputation_function_apply_set1.apply_set1_imputation(self.dataset)
        self.dataset = self.dataset[self.dataset['PERSON_AGE'].notna()]
        self.dataset.drop(self.dataset.loc[self.dataset['PERSON_SEX'] == "U"].index, inplace=True)
        self.dataset = self.dataset.dropna()

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

        '''
        df = self.dataset.copy()
        trnX, tstX, trnY, tstY = train_test_split(df, df['PERSON_INJURY'], test_size=0.3, random_state=1)
        balanceData = BalanceData(dataframe=pd.concat([trnX], axis=1))
        result_smote = balanceData.smote()
        trnX = result_smote.copy()
        trnY = result_smote['PERSON_INJURY']
        '''

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

    def scaleStandardScaler(self):
        transf = StandardScaler(with_mean=True, with_std=True, copy=True)
        self.dataset["PERSON_AGE"] = transf.fit_transform(self.dataset[["PERSON_AGE"]])

    def scaleMinMaxScaler(self):
        variables = ["BODILY_INJURY", "PERSON_TYPE", "EJECTION", "COMPLAINT", "PED_ROLE", "PED_ACTION",
                     "EMOTIONAL_STATUS", "SAFETY_EQUIPMENT"]

        for var in variables:
            transf = MinMaxScaler(feature_range=(0, 1), copy=True)
            self.dataset[var] = transf.fit_transform(self.dataset[[var]])

    def getOutliersWithLocalOutlierFactor(self):
        lof = LocalOutlierFactor()
        killed = self.dataset.loc[self.dataset["PERSON_INJURY"] == 'Injured']
        killed = killed.copy()
        yhat = lof.fit_predict(killed[[ 'SAFETY_EQUIPMENT', 'PED_ACTION', 'BODILY_INJURY', 'PERSON_TYPE', 'EJECTION','COMPLAINT','EMOTIONAL_STATUS','PED_ROLE']])
        # yhat = lof.fit(self.dataset[['PED_LOCATION', 'SAFETY_EQUIPMENT', 'PED_ACTION', 'BODILY_INJURY', 'PERSON_TYPE', 'EJECTION','COMPLAINT','EMOTIONAL_STATUS','POSITION_IN_VEHICLE','PED_ROLE']])

        mask = yhat != 1
        counter = 0
        for i in mask:
            if not i:
                counter += 1
        print("mask: ", mask)
        print("outliers tamanho: ", counter)
        print("Tamanho: ", len(mask))

        test = killed[mask]
        print(test)
        test.to_csv("data/{}.csv".format("outliersDataset"))


if __name__ == '__main__':
    pipeline = Pipeline()
    print("Initial pipeline shape: ", pipeline.dataset.shape)

    pipeline.removeIncorrectValues()

    pipeline.missing_values_imputation()

    pipeline.groupSafetyEquipment()
    pipeline.groupTime()
    pipeline.save_pipeline("grouped")

    pipeline.encode_BodilyInjury()
    pipeline.encode_PersonType()
    pipeline.encode_Ejection()
    pipeline.encode_Complaint()
    pipeline.encode_PedRole()
    pipeline.encode_PedAction()
    pipeline.encode_EmotionalStatus()
    pipeline.encode_SafetyEquipment()

    # pipeline.encode_PedLocationAndPositionInVehicle()

    pipeline.save_pipeline("encoded_notScaled")

    pipeline.getOutliersWithLocalOutlierFactor()

    pipeline.scaleStandardScaler()
    pipeline.scaleMinMaxScaler()

    pipeline.save_pipeline("encoded_scaled")

    pipeline.saveTrainAndTestData("test", "train")

    # print(pipeline.dataset[pipeline.dataset.isna().any(axis=1)])
