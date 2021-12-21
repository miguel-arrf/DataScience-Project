import pandas as pd
from imblearn.over_sampling import SMOTE
from pandas import read_csv, DataFrame, concat, Series
from matplotlib.pyplot import figure, savefig, show
from ds_charts import bar_chart, multiple_bar_chart, get_variable_types

class BalanceData():
    def __init__(self, dataframe):
        self.original = dataframe
        self.class_var = 'ALARM'
        target_count = self.original[self.class_var].value_counts()
        positive_class = target_count.idxmin()
        negative_class = target_count.idxmax()
        # ind_positive_class = target_count.index.get_loc(positive_class)
        self.df_positives = self.original[self.original[self.class_var] == positive_class]
        self.df_negatives = self.original[self.original[self.class_var] == negative_class]

        symbolic_vars = get_variable_types(self.original)['Symbolic']
        for symbolic_var in symbolic_vars:
            self.original[symbolic_var] = pd.factorize(self.original[symbolic_var])[0]

    def under_sampled(self):
        df_neg_sample = DataFrame(self.df_negatives.sample(len(self.df_positives)))
        df_under = concat([self.df_positives, df_neg_sample], axis=0)
        return df_under

    def pos_sample(self):
        df_pos_sample = DataFrame(self.df_positives.sample(len(self.df_negatives), replace=True))
        df_over = concat([df_pos_sample, self.df_negatives], axis=0)
        return df_over

    def smote(self):
        RANDOM_STATE = 42

        smote = SMOTE(sampling_strategy='minority', random_state=RANDOM_STATE)
        y = self.original.pop(self.class_var).values
        X = self.original.values
        smote_X, smote_y = smote.fit_resample(X, y)
        df_smote = concat([DataFrame(smote_X), DataFrame(smote_y)], axis=1)
        df_smote.columns = list(self.original.columns) + [self.class_var]
        return df_smote