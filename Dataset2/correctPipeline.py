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



class Pipeline:
    def __init__(self):
        filename = '../data/NYC_collisions_tabular.csv'
        self.dataset = read_csv(filename, index_col='UNIQUE_ID', na_values='', parse_dates=True,
                                infer_datetime_format=True)

        #TODO: SAMPLE DATA
        df_danger = self.dataset[self.dataset["ALARM"] == "Danger"]
        df_safe = self.dataset[self.dataset["ALARM"] == "Safe"]
        df_safe = df_safe.sample(frac=0.5, random_state=7)
        self.dataset = pd.concat([df_danger, df_safe], axis=0)

        self.dataset.drop(["COLLISION_ID"], axis=1, inplace=True)
        #TODO: Drop variáveis chatas

    def removeIncorrectValues(self):
        #TODO Incorrect values needs some love :D
        print("Incorrect values needs some love :D")

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


    def save_pipeline(self, datasetName):
        self.dataset.to_csv("data/{}.csv".format(datasetName))
        print("saved: {}, with shape:  {}".format(datasetName, self.dataset.shape))

    def dropMissingValues(self):
        dataset = self.dataset.copy()
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

    datasetDropMissingValues = pipeline.dropMissingValues()
    datasetReplaceMVByConstant = pipeline.replaceMissingValuesImputationWithConstant()
    datasetReplaceMVByMostCommon = pipeline.replaceMissingValuesImputationWithMostCommon()




    print("saved")
    results = []
    models = []
    for mv in [
        (datasetDropMissingValues, "dropMissingValues"),
        # (datasetReplaceMVByMostCommon, "ReplaceMVByMostCommon"),
        (datasetReplaceMVByConstant, "ReplaceMVByConstant")
    ]:

        print("____________")
        print("Missing value imputation method: ", mv[1])
        print()

        for dummyMethod in [
            # (encode, "ordinalEncoding"),
            (dummifyDataset, "oneHot"),
        ]:
            print("\tEncoding method: ", dummyMethod[1])

            for scalingMethod in [
                (scaleStandardScaler, "z-score"),
                # (scaleMinMaxScaler, "minMax"),
                (balanceNothing, "")
            ]:
                print("\tScaling method: ", scalingMethod[1])

                for balancingMethod in [
                    # (getBestUnderstampling, "bestUnderSampling"),
                    # (getUndersampling, "UnderSampling"),
                    # (getSmote, "Smote"),
                    # (getOverSampling, "OverSampling"),
                    (balanceNothing, "")
                ]:

                    print("\tBalancing method: ", balancingMethod[1])
                    for selectionFeatures in [
                        # (selectEverything, "", "-"),
                        # (select_redundant, "RedundantFeatures", 0.9),
                        # (select_redundant, "RedundantFeatures", 0.7),
                        # (drop_variance, "selectVariance", 0.9),
                        (balanceNothing, "noSelection", "")
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
                        '''

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

    print("Results:")
    for result in results:
        for key, value in result.items():
            print(key, " ", value)

        print()

