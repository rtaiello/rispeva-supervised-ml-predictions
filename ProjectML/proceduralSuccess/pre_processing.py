# std lib

# my lib
from ProjectML.general_util.imputation import *
from ProjectML.general_util.pre_processing import *
# third part
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import *
import pandas as pd
import numpy as np

# constant
LABEL = 'ProceduralSuccess'

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


def extract_label(dataset, label_name, label, percent):
    is_dead = dataset[dataset[label_name] == label]
    X_is_dead = is_dead.loc[:, 'CenterID':'P2Y12 inhibt']
    y_is_dead = is_dead.loc[:, label_name]
    X_is_dead_train, X_is_dead_test, y_is_dead_train, y_is_dead_test = train_test_split(
        X_is_dead, y_is_dead, test_size=percent, random_state=42)
    y_is_dead_test = pd.concat([X_is_dead_test, y_is_dead_test], axis=1, sort=False)
    dataset = dataset.drop(index=y_is_dead_test.index.tolist())
    return dataset, y_is_dead_test


def extract_test(dataset, label_name, percent_1, percent_0):
    dataset, dataset_dead = extract_label(dataset, label_name, 1, percent_1)
    dataset, dataset_alive = extract_label(dataset, label_name, 0, percent_0)
    return dataset, pd.concat([dataset_alive, dataset_dead])


def scaleColumns(df, cols_to_scale):
    for col in cols_to_scale:
        df[col] = pd.DataFrame(StandardScaler().fit_transform(pd.DataFrame(df[col])), columns=[col])
    return df


def imputation(dataset):
    label_miss = dataset[dataset[LABEL].isnull()].index.tolist()
    dataset = dataset.drop(index=label_miss)
    missing_values = dataset.isnull().sum(axis=1) > dataset.columns.size * 0.40
    missing_values = missing_values[missing_values == True]
    dataset = dataset.drop(index=missing_values.index.tolist())

    d_month_mean_null = dataset.isnull().mean() * 100
    col_removed = d_month_mean_null[d_month_mean_null >= 40.0].index.tolist()
    dataset = dataset.drop(columns=col_removed)

    binary_cols = [col for col in dataset if np.isin(dataset[col].dropna().unique(), [0, 1]).all()]
    dataset[binary_cols] = dataset[binary_cols].astype('bool')
    integer_cols = dataset.select_dtypes(include=['int64'])

    dataset = my_l_imp_KNN(dataset)
    dataset[binary_cols] = dataset[binary_cols].astype('bool')
    dataset[integer_cols.columns] = dataset[integer_cols.columns].astype('int64')
    return dataset, label_miss
