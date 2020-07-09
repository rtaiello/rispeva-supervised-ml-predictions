# std lib

# my lib
from ProjectML.general_util.pre_processing import *
from ProjectML.general_util.imputation import *

# third part
import numpy as np
# Constant
LABEL = '1monthDeath'


def read_dataset(file_path):
    dataset = my_l_read(file_path)
    X = dataset.loc[:, 'CenterID':'P2Y12inhibt']
    y = dataset.loc[:, LABEL]
    return pd.concat([X, y], axis=1, sort=False)


def extract_feature(dataset):
    X = dataset.loc[:, 'CenterID':'P2Y12inhibt']
    y = dataset.loc[:, LABEL]
    X = my_l_rm_white_space(X)
    # X = X.drop(columns=['CenterID', 'PatientID'])
    return X, y, pd.concat([X, y], axis=1, sort=False)


def imputation(dataset):
    # didn't remove the LABEL

    # removed feature with 60% of NaN
    d_month_mean_null = dataset.isnull().mean() * 100
    col_removed = d_month_mean_null[d_month_mean_null >= 60.0].index.tolist()
    dataset = dataset.drop(columns=col_removed)

    binary_cols = [col for col in dataset if np.isin(dataset[col].dropna().unique(), [0, 1]).all()]
    dataset[binary_cols] = dataset[binary_cols].astype('bool')
    interger_cols = dataset.select_dtypes(include=['int64'])
    dataset,imputer = my_l_imp_MICE(dataset)
    dataset[binary_cols] = dataset[binary_cols].round()
    dataset[binary_cols] = dataset[binary_cols].astype('bool')
    dataset[interger_cols.columns] = dataset[interger_cols.columns].astype('int64')
    return dataset, col_removed, imputer
