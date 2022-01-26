import numpy as np
import pandas as pd
import xlrd
from imblearn.over_sampling import SMOTE
from matplotlib.pyplot import show, savefig
from numpy import nan, unique
from pandas import DataFrame, concat
from pandas import read_csv
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import *

from Dataset1.DecisionTrees import DecisionTrees
from Dataset1.Gradientboosting import GradientBoosting
from Dataset1.MLP import MLP
from Dataset1.MissingValuesImputation_Set1 import imputation_function_apply_set1
from Dataset1.NaiveBayes import NaiveBayes
from Dataset1.KNN import KNN
from Dataset1.RandomForests import RandomForests

from ds_charts import get_variable_types, plot_evaluation_results

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

'''
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
'''


class Pipeline:
    def __init__(self):
        filename = '../data/NYC_collisions_tabular.csv'
        self.dataset = read_csv(filename, index_col='UNIQUE_ID', na_values='', parse_dates=True,
                                infer_datetime_format=True)

        self.dataset.drop(["COLLISION_ID"], axis=1, inplace=True)
        self.dataset.drop(["PERSON_ID"], axis=1, inplace=True)
        self.dataset.drop(["VEHICLE_ID"], axis=1, inplace=True)
        self.dataset.drop(["CRASH_TIME"], axis=1, inplace=True)
        self.dataset.drop(["CRASH_DATE"], axis=1, inplace=True)

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

        # self.dataset['CRASH_TIME'] = self.dataset['CRASH_TIME'].apply(
        #    lambda x: transform_time(x))

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
            imp = SimpleImputer(strategy='constant', fill_value='Unknown', missing_values=nan, copy=True)
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


def encode_PosInVehicle(dataset):
    dataset = dataset.copy()


def encode_EJECTION_LOCATION_ROLE_TYPE(dataset):
    encode_ejection = [("Not Ejected", 5), ("Trapped", 7), ("Ejected", 10), ("Partially Ejected", 8), ("Unknown", 0)]
    encode_location = {"Does Not Apply": 3, "Pedestrian/Bicyclist/Other Pedestrian Not at Intersection": 4,
                       "Pedestrian/Bicyclist/Other Pedestrian at Intersection": 5, "Unknown": 1}
    encode_role = {"Driver": 5, "Pedestrian": 4, "Passenger": 3, "Other": 1, "In-Line Skater": 2}
    encode_type = {"Occupant": 2, "Pedestrian": 3, "Bicyclist": 1, "Other Motorized": 4}

    for value in encode_ejection:
        dataset["EJECTION"].loc[(dataset["EJECTION"] == value[0])] = value[1]

    for key in encode_location:
        dataset["PED_LOCATION"].loc[(dataset["PED_LOCATION"] == key)] = encode_location[key]

    for value in encode_role:
        dataset["PED_ROLE"].loc[(dataset["PED_ROLE"] == value)] = encode_role[value]

    for value in encode_type:
        dataset["PERSON_TYPE"].loc[(dataset["PERSON_TYPE"] == value)] = encode_type[value]


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
    # dataset_copy = encode_PosInVehicle(dataset_copy)
    # dataset_copy = encode_BodilyInjury(dataset_copy)
    # dataset_copy = encode_PersonType(dataset_copy)
    # dataset_copy = encode_Ejection(dataset_copy)
    # dataset_copy = encode_Complaint(dataset_copy)
    # dataset_copy = encode_PedRole(dataset_copy)
    # dataset_copy = encode_PedAction(dataset_copy)
    # dataset_copy = encode_EmotionalStatus(dataset_copy)
    # dataset_copy = encode_SafetyEquipment(dataset_copy)

    # for symbolic_var in get_variable_types(dataset_copy)['Symbolic']:
    #    dataset_copy = encode_by_order(dataset_copy, symbolic_var)

    workbook = xlrd.open_workbook(f'encoding.xlsx', on_demand=True)
    sheetNames = workbook.sheet_names()
    encoding = []
    for sheet in range(len(sheetNames)):

        worksheet = workbook.sheet_by_index(sheet)

        first_row = []  # The row where we stock the name of the column
        for col in range(worksheet.ncols):
            first_row.append(worksheet.cell_value(0, col))
        # transform the workbook to a list of dictionaries
        data = []
        for row in range(1, worksheet.nrows):
            elm = {}
            for col in range(worksheet.ncols):
                elm[first_row[col]] = worksheet.cell_value(row, col)
            data.append(elm)
        encoding.append(data)

        for value in data:
            if sheetNames[sheet] != "EMOTIONAL_STATUS":
                dataset_copy[sheetNames[sheet]].loc[(dataset_copy[sheetNames[sheet]] == value['Name'])] = value['Value']

    encode_EJECTION_LOCATION_ROLE_TYPE(dataset_copy)

    # for symbolic_var in ["CRASH_TIME", "CRASH_DATE"]:
    #    dataset_copy = encode_by_order(dataset_copy, symbolic_var)
    dataset_copy["PERSON_SEX"].replace(('F', 'M'), (1, 0), inplace=True)
    return dataset_copy


def saveTrainAndTestData(dataset):
    df = dataset.copy()
    y = df['PERSON_INJURY']
    X = df.drop(["PERSON_INJURY"], axis=1)
    return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


def getUndersampling(dataset):
    original = dataset.copy()

    class_var = 'PERSON_INJURY'
    target_count = original[class_var].value_counts()

    positive_class = target_count.idxmin()
    negative_class = target_count.idxmax()

    df_positives = original[original[class_var] == positive_class]
    df_negatives = original[original[class_var] == negative_class]
    df_neg_sample = DataFrame(df_negatives.sample(len(df_positives)))
    df_under = concat([df_positives, df_neg_sample], axis=0)
    return df_under


def getSmote(dataset):
    original = dataset.copy()
    class_var = 'PERSON_INJURY'

    RANDOM_STATE = 42

    smote = SMOTE(sampling_strategy='minority', random_state=RANDOM_STATE)
    y = original.pop(class_var).values
    X = original.values
    smote_X, smote_y = smote.fit_resample(X, y)
    df_smote = concat([DataFrame(smote_X), DataFrame(smote_y)], axis=1)
    df_smote.columns = list(original.columns) + [class_var]
    return df_smote


def getBestUnderstampling(dataset):
    original = dataset.copy()

    class_var = 'PERSON_INJURY'
    target_count = original[class_var].value_counts()

    positive_class = target_count.idxmin()
    negative_class = target_count.idxmax()

    df_positives = original[original[class_var] == positive_class]
    df_negatives = original[original[class_var] == negative_class]
    df_negatives = df_negatives.sample(15000)  # Sampled df_negatives
    # df_neg_sample = DataFrame(df_negatives.sample(len(df_positives)))
    df_under = concat([df_positives, df_negatives], axis=0)
    return getSmote(df_under)


def getOverSampling(dataset):
    original = dataset.copy()
    class_var = 'PERSON_INJURY'

    target_count = original[class_var].value_counts()

    positive_class = target_count.idxmin()
    negative_class = target_count.idxmax()

    df_positives = original[original[class_var] == positive_class]
    df_negatives = original[original[class_var] == negative_class]

    df_pos_sample = DataFrame(df_positives.sample(len(df_negatives), replace=True))
    df_over = concat([df_pos_sample, df_negatives], axis=0)
    return df_over


def trainKNN(trainX, testX, trainY, testY, model):
    estimators = {'KNN': KNeighborsClassifier(n_neighbors=5, metric='euclidean')}

    labels = unique(trainY)
    labels.sort()

    for clf in estimators:
        estimators[clf].fit(trainX, trainY)

        prd_trn = estimators[clf].predict(trainX)
        prd_tst = estimators[clf].predict(testX)
        plot_evaluation_results(labels, trainY, prd_trn, testY, prd_tst, extra=f"for {clf}-{model}")
        savefig(f"results/{clf}-{model}.png")
        show()
        print("\t\tprecision value for {}: ".format(clf), precision_score(testY, prd_tst, pos_label="Killed"))
        print("\t\trecall value for {}: ".format(clf), recall_score(testY, prd_tst, pos_label="Killed"))
        print()

        return precision_score(testY, prd_tst, pos_label="Killed"), recall_score(testY, prd_tst, pos_label="Killed")


def trainNaiveBayes(trainX, testX, trainY, testY, model):
    estimators = {'GaussianNB': GaussianNB()}

    labels = unique(trainY)
    labels.sort()

    for clf in estimators:
        estimators[clf].fit(trainX, trainY)

        prd_trn = estimators[clf].predict(trainX)
        prd_tst = estimators[clf].predict(testX)
        plot_evaluation_results(labels, trainY, prd_trn, testY, prd_tst, extra=f"for {clf}-{model}")
        savefig(f"results/{clf}-{model}.png")
        show()
        print("\t\tprecision value for {}: ".format(clf), precision_score(testY, prd_tst, pos_label="Killed"))
        print("\t\trecall value for {}: ".format(clf), recall_score(testY, prd_tst, pos_label="Killed"))
        print()

        return precision_score(testY, prd_tst, pos_label="Killed"), recall_score(testY, prd_tst, pos_label="Killed")


def scaleStandardScaler(dataset):
    dataset = dataset.copy()

    variable_types = get_variable_types(dataset)
    numeric_vars = variable_types['Numeric']
    symbolic_vars = variable_types['Symbolic']
    boolean_vars = variable_types['Binary']

    df_nr = dataset[numeric_vars]
    df_sb = dataset[symbolic_vars]
    df_bool = dataset[boolean_vars]

    transf = StandardScaler(with_mean=True, with_std=True, copy=True).fit(df_nr)
    tmp = DataFrame(transf.transform(df_nr), index=dataset.index, columns=numeric_vars)
    norm_data_zscore = concat([tmp, df_sb, df_bool], axis=1)
    return norm_data_zscore


def scaleMinMaxScaler(dataset):
    dataset = dataset.copy()

    variable_types = get_variable_types(dataset)
    numeric_vars = variable_types['Numeric']
    symbolic_vars = variable_types['Symbolic']
    boolean_vars = variable_types['Binary']

    df_nr = dataset[numeric_vars]
    df_sb = dataset[symbolic_vars]
    df_bool = dataset[boolean_vars]

    transf = MinMaxScaler(feature_range=(0, 1), copy=True).fit(df_nr)
    tmp = DataFrame(transf.transform(df_nr), index=dataset.index, columns=numeric_vars)
    norm_data_minmax = concat([tmp, df_sb, df_bool], axis=1)
    return norm_data_minmax


def balanceNothing(dataset):
    return dataset.copy()


def selectEverything(dataset, threshold):
    return dataset.copy()


def drop_variance(originalDataset, threshold):
    dataset = originalDataset.copy()
    injury = dataset["PERSON_INJURY"]
    dataset = dataset.drop(["PERSON_INJURY"], axis=1)
    dataset = dataset.apply(pd.to_numeric)

    numeric = get_variable_types(dataset)['Numeric']
    dataset = dataset[numeric]

    lst_variables = []
    lst_variances = []
    for el in dataset.columns:
        value = dataset[el].var()
        if value >= threshold:
            lst_variables.append(el)
            lst_variances.append(value)

    print("\tvars to drop: {}: {}".format(len(lst_variables), lst_variables))
    print("original columns: ", len(dataset.columns))
    dataset = dataset.drop(lst_variables, axis=1)

    return pd.concat([dataset, injury], axis=1)


def drop_redundant(data: DataFrame, vars_2drop: dict) -> DataFrame:
    sel_2drop = []
    for key in vars_2drop.keys():
        if key not in sel_2drop:
            for r in vars_2drop[key]:
                if r != key and r not in sel_2drop:
                    sel_2drop.append(r)
    print('\tVariables to drop', sel_2drop)
    df = data.copy()
    for var in sel_2drop:
        df.drop(labels=var, axis=1, inplace=True)

    return df


def select_redundant(originalDataset, threshold):
    dataset = originalDataset.copy()
    injury = dataset["PERSON_INJURY"]
    dataset = dataset.drop(["PERSON_INJURY"], axis=1)
    dataset = dataset.apply(pd.to_numeric)

    corr_mtx = dataset.corr()

    if corr_mtx.empty:
        return {}

    corr_mtx = abs(corr_mtx)
    vars_2drop = {}
    for el in corr_mtx.columns:
        el_corr = (corr_mtx[el]).loc[corr_mtx[el] >= threshold]
        if len(el_corr) == 1:
            corr_mtx.drop(labels=el, axis=1, inplace=True)
            corr_mtx.drop(labels=el, axis=0, inplace=True)
        else:
            vars_2drop[el] = el_corr.index

    # dataset = dataset.drop(vars_2drop.keys(), axis=1)
    # print("Vars 2 drop with threshold: {} -> {}".format(threshold, vars_2drop.keys()))
    dataset = pd.concat([dataset, injury], axis=1)

    return drop_redundant(dataset, vars_2drop)


if __name__ == '__main__':
    pipeline = Pipeline()
    print("Initial pipeline shape: ", pipeline.dataset.shape)

    pipeline.removeIncorrectValues()

    # pipeline.groupSafetyEquipment()
    # pipeline.groupTime()

    pipeline.dataset = pipeline.dataset.drop(["EMOTIONAL_STATUS"], axis=1)

    datasetCustomImputation = pipeline.customMissingValuesImputation()
    datasetReplaceMVByConstant = pipeline.replaceMissingValuesImputationWithConstant()
    datasetReplaceMVByMostCommon = pipeline.replaceMissingValuesImputationWithMostCommon()

    select_redundant(scaleStandardScaler(encode(datasetCustomImputation)), 0.7).to_csv("teste_to_use.csv")
    scaleStandardScaler(dummifyDataset(datasetCustomImputation)).to_csv("teste_to_use_encoded.csv")

    print("saved")
    results = []
    models = []
    for mv in [
        (datasetCustomImputation, "customImputation"),
        #(datasetReplaceMVByMostCommon, "ReplaceMVByMostCommon"),
        #(datasetReplaceMVByConstant, "ReplaceMVByConstant")
    ]:

        print("____________")
        print("Missing value imputation method: ", mv[1])
        print()

        for dummyMethod in [
            (encode, "ordinalEncoding"),
            #(dummifyDataset, "oneHot"),
        ]:
            print("\tEncoding method: ", dummyMethod[1])

            for scalingMethod in [
                (scaleStandardScaler, "z-score"),
                #(scaleMinMaxScaler, "minMax"),
                #(balanceNothing, "")
            ]:
                print("\tScaling method: ", scalingMethod[1])

                for balancingMethod in [
                                       (getBestUnderstampling, "bestUnderSampling"),
                                         #(getUndersampling, "UnderSampling"),
                                         #(getSmote, "Smote"),
                                         #(getOverSampling, "OverSampling"),
                                       # (balanceNothing, "")
                                        ]:

                    print("\tBalancing method: ", balancingMethod[1])
                    for selectionFeatures in [
                        #(selectEverything, "", "-"),
                        #(select_redundant, "RedundantFeatures", 0.9),
                        (select_redundant, "RedundantFeatures", 0.7),
                        #(drop_variance, "selectVariance", 0.9),
                        #(balanceNothing, "noSelection", "")
                    ]:
                        print("\tfeatureSelection: ", selectionFeatures[1])

                        mvDataset = mv[0].copy()
                        dummified = dummyMethod[0](mvDataset)
                        dummified = scalingMethod[0](dummified)
                        dummified = selectionFeatures[0](dummified, selectionFeatures[2])

                        print("")

                        if "PERSON_SEX" in dummified.columns:
                            dummified["PERSON_SEX"].replace(('F', 'M'), (1, 0), inplace=True)

                        X_train, X_test, y_train, y_test = saveTrainAndTestData(dummified)

                        temp = pd.concat([X_train, y_train], axis=1)
                        temp = balancingMethod[0](temp)
                        y_train = temp["PERSON_INJURY"]
                        X_train = temp.drop(["PERSON_INJURY"], axis=1)

                        randomForests = RandomForests(trnX=X_train,
                                                      trnY=y_train,
                                                      tstX=X_test,
                                                      tstY=y_test)
                        # Classification section:
                        '''
                        naiveBayes = NaiveBayes( trnX=X_train,
                                                trnY=y_train,
                                                tstX=X_test,
                                                tstY=y_test)
                        
                        
                        knn = KNN(trnX=X_train,
                                  trnY=y_train,
                                  tstX=X_test,
                                  tstY=y_test)
                        

                        decisionTrees = DecisionTrees(trnX=X_train,
                                                      trnY=y_train,
                                                      tstX=X_test,
                                                      tstY=y_test)
                        randomForests = RandomForests(trnX=X_train,
                                                      trnY=y_train,
                                                      tstX=X_test,
                                                      tstY=y_test)
                        
                        mlp = MLP(trnX=X_train,
                                  trnY=y_train,
                                  tstX=X_test,
                                  tstY=y_test)

                        gradientBoosting = GradientBoosting(trnX=X_train,
                                  trnY=y_train,
                                  tstX=X_test,
                                  tstY=y_test)
                        



                        # Methods evaluation:
                        dicionary = {}

                        prNB, recallNB = trainNaiveBayes(X_train, X_test, y_train, y_test,
                                                         model=f"{dummyMethod[1]}-{mv[1]}-{scalingMethod[1]}-{balancingMethod[1]}-{selectionFeatures[1]}-{selectionFeatures[2]}")
                        dicionary[
                            f"{mv[1]}-{dummyMethod[1]}-naiveBayes-{scalingMethod[1]}-{balancingMethod[1]}-{selectionFeatures[1]}-{selectionFeatures[2]}"] = "precision:{:.2f} <-> recall:{:.2f}".format(
                            prNB, recallNB)

                        prKNN, recallKNN = trainKNN(X_train, X_test, y_train, y_test,
                                                    model=f"{dummyMethod[1]}-{mv[1]}-{scalingMethod[1]}-{balancingMethod[1]}-{selectionFeatures[1]}-{selectionFeatures[2]}")
                        dicionary[
                            f"{mv[1]}-{dummyMethod[1]}-KNN-{scalingMethod[1]}-{balancingMethod[1]}-{selectionFeatures[1]}-{selectionFeatures[2]}"] = "precision:{:.2f} <-> recall:{:.2f} ".format(
                            prKNN, recallKNN)

                        results.append(dicionary)
                        '''


    print("Results:")
    for result in results:
        for key, value in result.items():
            print(key, " ", value)

        print()

