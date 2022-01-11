import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from matplotlib.pyplot import show
from numpy import nan, unique
from pandas import DataFrame, concat
from pandas import read_csv
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import cross_val_score, KFold, cross_val_predict, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import *

from Dataset1.MissingValuesImputation_Set1 import imputation_function_apply_set1
from ds_charts import get_variable_types, plot_evaluation_results

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def smote(train):
    RANDOM_STATE = 42
    class_var = 'PERSON_INJURY'

    smoted = SMOTE(sampling_strategy='minority', random_state=RANDOM_STATE, n_jobs=-1)
    y = train.pop(class_var).values
    X = train.values

    smote_X, smote_y = smoted.fit_resample(X, y)
    df_smote = pd.concat([DataFrame(smote_X), DataFrame(smote_y)], axis=1)
    df_smote.columns = list(train.columns) + [class_var]
    return df_smote


class Pipeline:
    def __init__(self):
        filename = '../data/NYC_collisions_tabular.csv'
        self.dataset = read_csv(filename, index_col='UNIQUE_ID', na_values='', parse_dates=True,
                                infer_datetime_format=True)

        self.dataset.drop(["COLLISION_ID"], axis=1, inplace=True)
        self.dataset.drop(["PERSON_ID"], axis=1, inplace=True)
        self.dataset.drop(["VEHICLE_ID"], axis=1, inplace=True)

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
        self.dataset.drop(self.dataset.loc[self.dataset['PERSON_SEX'] == "U"].index, inplace=True)

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
            if isinstance(v, float):
                return v
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

    def customMissingValuesImputation(self):
        dataset = self.dataset.copy()

        imputation_function_apply_set1.apply_set1_imputation(dataset)
        dataset = dataset[dataset['PERSON_AGE'].notna()]
        dataset.drop(dataset.loc[dataset['PERSON_SEX'] == "U"].index, inplace=True)
        dataset = dataset.dropna()

        return dataset

    def dropAllMissingValues(self):
        dataset = self.dataset.copy()
        return dataset.dropna()

    def replaceMissingValuesImputationWithConstant(self):
        dataset = self.dataset.copy()

        tmp_nr, tmp_sb, tmp_bool = None, None, None
        variables = get_variable_types(dataset)
        numeric_vars = variables['Numeric']
        symbolic_vars = variables['Symbolic']
        binary_vars = variables['Binary']
        if len(numeric_vars) > 0:
            imp = SimpleImputer(strategy='constant', fill_value=0, missing_values=nan, copy=True)
            tmp_nr = DataFrame(imp.fit_transform(dataset[numeric_vars]), columns=numeric_vars)
        if len(symbolic_vars) > 0:
            imp = SimpleImputer(strategy='constant', fill_value='NA', missing_values=nan, copy=True)
            tmp_sb = DataFrame(imp.fit_transform(dataset[symbolic_vars]), columns=symbolic_vars)
        if len(binary_vars) > 0:
            imp = SimpleImputer(strategy='constant', fill_value=False, missing_values=nan, copy=True)
            tmp_bool = DataFrame(imp.fit_transform(dataset[binary_vars]), columns=binary_vars)

        df = concat([tmp_nr, tmp_sb, tmp_bool], axis=1)
        return df

    def replaceMissingValuesImputationWithMostCommon(self):
        dataset = self.dataset.copy()

        tmp_nr, tmp_sb, tmp_bool = None, None, None
        variables = get_variable_types(dataset)
        numeric_vars = variables['Numeric']
        symbolic_vars = variables['Symbolic']
        binary_vars = variables['Binary']

        if len(numeric_vars) > 0:
            imp = SimpleImputer(strategy='mean', fill_value=0, missing_values=nan, copy=True)
            tmp_nr = DataFrame(imp.fit_transform(dataset[numeric_vars]), columns=numeric_vars)
        if len(symbolic_vars) > 0:
            imp = SimpleImputer(strategy='most_frequent', fill_value='NA', missing_values=nan, copy=True)
            tmp_sb = DataFrame(imp.fit_transform(dataset[symbolic_vars]), columns=symbolic_vars)
        if len(binary_vars) > 0:
            imp = SimpleImputer(strategy='constant', fill_value=False, missing_values=nan, copy=True)
            tmp_bool = DataFrame(imp.fit_transform(dataset[binary_vars]), columns=binary_vars)

        df = concat([tmp_nr, tmp_sb, tmp_bool], axis=1)
        return df


def dummifyDataset(data):
    dataset = data.copy()
    variables = get_variable_types(dataset)
    symbolic_vars = variables['Symbolic']
    df = dummify(dataset, symbolic_vars)
    return df


def encode_BodilyInjury(dataset):
    dataset = dataset.copy()

    categories_dict = dataset['BODILY_INJURY'].value_counts().to_dict()

    sortedCategories = []
    for key in categories_dict:
        sortedCategories.append(key)

    sortedCategories = [np.array(list(reversed(sortedCategories)))]

    bodilyInjuryEncoder = OrdinalEncoder(categories=sortedCategories)
    dataset["BODILY_INJURY"] = bodilyInjuryEncoder.fit_transform(dataset[["BODILY_INJURY"]])
    return dataset


def encode_by_order(dataset, var):
    dataset = dataset.copy()

    categories_dict = dataset[var].value_counts().to_dict()

    sortedCategories = []
    for key in categories_dict:
        sortedCategories.append(key)

    sortedCategories = [np.array(list(reversed(sortedCategories)))]

    encoder = OrdinalEncoder(categories=sortedCategories)
    dataset[var] = encoder.fit_transform(dataset[[var]])
    return dataset


def encode_PedAction(dataset):
    dataset = dataset.copy()

    categories = dataset["PED_ACTION"].unique()
    sortedCategories = ["Crossing,No Signal,or Crosswalk", "Crossing With Signal", "Other Actions in Roadway"]
    for category in categories:
        if category not in sortedCategories:
            sortedCategories.append(category)

    sortedCategories = [np.array(list(reversed(sortedCategories)))]
    pedActionEncoder = OrdinalEncoder(categories=sortedCategories)
    dataset["PED_ACTION"] = pedActionEncoder.fit_transform(dataset[["PED_ACTION"]])
    return dataset


def encode_PedRole(dataset):
    dataset = dataset.copy()
    categories = dataset["PED_ROLE"].unique()
    sortedCategories = ["Pedestrian", "Passanger", "Driver"]
    for category in categories:
        if category not in sortedCategories:
            sortedCategories.append(category)

    sortedCategories = [np.array(list(reversed(sortedCategories)))]
    pedRoleEncoder = OrdinalEncoder(categories=sortedCategories)
    dataset["PED_ROLE"] = pedRoleEncoder.fit_transform(dataset[["PED_ROLE"]])
    return dataset


def encode_EmotionalStatus(dataset):
    dataset = dataset.copy()
    categories = dataset["EMOTIONAL_STATUS"].unique()
    sortedCategories = ["Apparent Death", "Unconscious", "Conscious"]
    for category in categories:
        if category not in sortedCategories:
            sortedCategories.append(category)

    sortedCategories = [np.array(list(reversed(sortedCategories)))]
    emotionalStatusEncoder = OrdinalEncoder(categories=sortedCategories)
    dataset["EMOTIONAL_STATUS"] = emotionalStatusEncoder.fit_transform(dataset[["EMOTIONAL_STATUS"]])
    return dataset


def encode_Complaint(dataset):
    dataset = dataset.copy()
    categories = dataset["COMPLAINT"].unique()
    sortedCategories = ["Internal", "Complaint of Pain or Nausea",
                        "None Visible",
                        "Crush Injuries",
                        "Contusion - Bruise",
                        "Severe Bleeding"]
    for category in categories:
        if category not in sortedCategories:
            sortedCategories.append(category)

    sortedCategories = [np.array(list(reversed(sortedCategories)))]
    complaintEncoder = OrdinalEncoder(categories=sortedCategories)
    dataset["COMPLAINT"] = complaintEncoder.fit_transform(dataset[["COMPLAINT"]])
    return dataset


def encode_Ejection(dataset):
    dataset = dataset.copy()
    categories = dataset["EJECTION"].unique()
    sortedCategories = ["Ejected", "Not Ejected"]
    for category in categories:
        if category not in sortedCategories:
            sortedCategories.append(category)

    sortedCategories = [np.array(list(reversed(sortedCategories)))]
    ejectionEncoder = OrdinalEncoder(categories=sortedCategories)
    dataset["EJECTION"] = ejectionEncoder.fit_transform(dataset[["EJECTION"]])
    return dataset


def encode_SafetyEquipment(dataset):
    dataset = dataset.copy()
    categories = dataset["SAFETY_EQUIPMENT"].unique()
    sortedCategories = ["None", "Helmet", "Unknown", "AirBag", "Belt"]
    for category in categories:
        if category not in sortedCategories:
            sortedCategories.append(category)

    sortedCategories = [np.array(list(reversed(sortedCategories)))]
    safetyEquipmentEncoder = OrdinalEncoder(categories=sortedCategories)
    dataset["SAFETY_EQUIPMENT"] = safetyEquipmentEncoder.fit_transform(dataset[["SAFETY_EQUIPMENT"]])
    return dataset


def encode_PersonType(dataset):
    dataset = dataset.copy()
    categories = dataset["PERSON_TYPE"].unique()
    sortedCategories = ["Occupant", "Pedestrian"]
    for category in categories:
        if category not in sortedCategories:
            sortedCategories.append(category)

    sortedCategories = [np.array(list(reversed(sortedCategories)))]
    personTypeEncoder = OrdinalEncoder(categories=sortedCategories)
    dataset["PERSON_TYPE"] = personTypeEncoder.fit_transform(dataset[["PERSON_TYPE"]])
    return dataset


def dummify(df, vars_to_dummify):
    df = df.copy()
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


def encode(dataset):
    dataset_copy = dataset.copy()
    # dataset_copy = encode_BodilyInjury(dataset_copy)
    # dataset_copy = encode_PersonType(dataset_copy)
    # dataset_copy = encode_Ejection(dataset_copy)
    # dataset_copy = encode_Complaint(dataset_copy)
    # dataset_copy = encode_PedRole(dataset_copy)
    # dataset_copy = encode_PedAction(dataset_copy)
    # dataset_copy = encode_EmotionalStatus(dataset_copy)
    # dataset_copy = encode_SafetyEquipment(dataset_copy)

    for symbolic_var in get_variable_types(dataset_copy)['Symbolic']:
        dataset_copy = encode_by_order(dataset_copy, symbolic_var)

    return dataset_copy


def saveTrainAndTestData(dataset):
    '''
    df = dataset.copy()

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

    return smote(train), test
    '''
    df = dataset.copy()
    y = df['PERSON_INJURY']
    X = df.drop(["PERSON_INJURY"], axis=1)
    return train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)


def trainKNN(trainX, testX, trainY, testY):

    estimators = {'NaiveBayes': KNeighborsClassifier(n_neighbors=5, metric='manhattan')}

    labels = unique(trainY)
    labels.sort()

    for clf in estimators:
        estimators[clf].fit(trainX, trainY)

        prd_trn = estimators[clf].predict(trainX)
        prd_tst = estimators[clf].predict(testX)
        plot_evaluation_results(labels, trainY, prd_trn, testY, prd_tst)
        show()
        print("\tprecision value for {}: ".format(clf), precision_score(testY, prd_tst, pos_label="Killed"))
        print("\trecall value for {}: ".format(clf), recall_score(testY, prd_tst, pos_label="Killed"))


def trainNaiveBayes(trainX, testX, trainY, testY):

    estimators = {'GaussianNB': GaussianNB()}

    labels = unique(trainY)
    labels.sort()

    for clf in estimators:
        estimators[clf].fit(trainX, trainY)

        prd_trn = estimators[clf].predict(trainX)
        prd_tst = estimators[clf].predict(testX)
        plot_evaluation_results(labels, trainY, prd_trn, testY, prd_tst)
        show()
        print("\tprecision value for {}: ".format(clf), precision_score(testY, prd_tst, pos_label="Killed"))
        print("\trecall value for {}: ".format(clf), recall_score(testY, prd_tst, pos_label="Killed"))


def scaleStandardScaler(dataset):
    dataset = dataset.copy()
    transf = StandardScaler(with_mean=True, with_std=True, copy=True)
    dataset["PERSON_AGE"] = transf.fit_transform(dataset[["PERSON_AGE"]])
    return dataset


def scaleMinMaxScaler(dataset):
    dataset = dataset.copy()

    variables = ["BODILY_INJURY", "PERSON_TYPE", "EJECTION", "COMPLAINT", "PED_ROLE", "PED_ACTION",
                 "EMOTIONAL_STATUS", "SAFETY_EQUIPMENT"]

    for var in variables:
        transf = MinMaxScaler(feature_range=(0, 1), copy=True)
        dataset[var] = transf.fit_transform(dataset[[var]])
    return dataset


if __name__ == '__main__':
    pipeline = Pipeline()
    print("Initial pipeline shape: ", pipeline.dataset.shape)

    pipeline.removeIncorrectValues()

    pipeline.groupSafetyEquipment()
    pipeline.groupTime()

    # datasetRemovedMissingValues = pipeline.dropAllMissingValues()
    datasetCustomImputation = pipeline.customMissingValuesImputation()
    datasetReplaceMVByConstant = pipeline.replaceMissingValuesImputationWithConstant()
    datasetReplaceMVByMostCommon = pipeline.replaceMissingValuesImputationWithMostCommon()

    encode(datasetCustomImputation).to_csv("data/for_tree_labs.csv")

    for mv in [
        # DON'T USE:  (datasetRemovedMissingValues, "datasetRemovedMissingValues"),
        (datasetCustomImputation, "datasetCustomImputation"),
        (datasetReplaceMVByConstant, "datasetReplaceMVByConstant"),
        (datasetReplaceMVByMostCommon, "datasetReplaceMVByMostCommon")
    ]:

        print("____________")
        print("Missing value imputation method: ", mv[1])
        print()
        for dummyMethod in [
            (encode, "ordinal-encoding"),
            (dummifyDataset, "one-hot"),
        ]:
            print("\tEncoding method: ", dummyMethod[1])
            mvDataset = mv[0].copy()
            dummified = dummyMethod[0](mvDataset)
            print("\tresult size: ", dummified.shape)
            print("")

            dummified["PERSON_SEX"].replace(('F', 'M'), (1, 0), inplace=True)

            X_train, X_test, y_train, y_test = saveTrainAndTestData(dummified)

            trainNaiveBayes(X_train, X_test, y_train, y_test)
            trainKNN(X_train, X_test, y_train, y_test)

    # pipeline.saveTrainAndTestData("test", "train")
