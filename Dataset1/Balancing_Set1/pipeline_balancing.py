import numpy as np
import pandas as pd


def apply_balancing(df):
    class_var = 'PERSON_INJURY'
    target_count = df[class_var].value_counts()

    positive_class = target_count.idxmin()
    negative_class = target_count.idxmax()

    df_positives = df[df[class_var] == positive_class]
    df_negatives = df[df[class_var] == negative_class]

    df_positives = df_positives.sample(frac=1, random_state=7)
    df_negatives = df_negatives.sample(frac=1, random_state=7)


    a, b, c = np.split(df_positives, [int(.3 * len(df_positives)), int(.7 * len(df_positives))])
    positiveTest = a
    positiveTrain = pd.concat([b,c], axis=0)
    # positiveTest, positiveTrain = np.split(df_positives, [int(.3 * len(df_positives)), int(.7 * len(df_positives))])

    a_neg, b_neg, c_neg = np.split(df_negatives, [int(.3 * len(df_negatives)), int(.7 * len(df_negatives))])
    negativeTest = a_neg
    negativeTrain = pd.concat([b_neg, c_neg], axis=0)
    # negativeTest, negativeTrain = np.split(df_negatives, [int(.3 * len(df_negatives)), int(.7 * len(df_negatives))])

    train_temp = pd.concat([positiveTrain, negativeTrain], axis=0)
    test_temp = pd.concat([positiveTest, negativeTest], axis=0)

    X_train = train_temp.drop(class_var, axis=1)
    y_train = train_temp[class_var]
    X_test = test_temp.drop(class_var, axis=1)
    y_test = test_temp[class_var]

    print("X_train: {}; y_train: {}; X_test: {}, y_test: {}.".format(
        X_train.shape,
        y_train.shape,
        X_test.shape,
        y_test.shape,
    ))

    return X_train, X_test, y_train, y_test
