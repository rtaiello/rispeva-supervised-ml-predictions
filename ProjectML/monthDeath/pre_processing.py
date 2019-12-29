# std lib

# my lib
from ProjectML.general_util import my_l_imp, my_l_rebalance
# third part
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


def imputation(dataset, lab):
    row_removed = dataset[dataset['1 month Death'].isnull()].index.tolist()
    dataset = dataset.drop(index=row_removed)

    d_month_mean_null = dataset.isnull().mean() * 100
    col_removed = d_month_mean_null[d_month_mean_null >= 60.0].index.tolist()
    dataset = dataset.drop(columns=col_removed)

    binary_cols = [col for col in dataset if np.isin(dataset[col].dropna().unique(), [0.0, 1.0]).all()]

    dataset = my_l_imp("KNN", dataset)[0]
    print(dataset[dataset.isna().any(axis=1)])
    dataset[binary_cols] = dataset[binary_cols].round()

    return dataset, row_removed


def extract_label(dataset,label,percent):
    is_dead = dataset[dataset['1 month Death'] == label]
    X_is_dead = is_dead.loc[:, 'CenterID':'P2Y12 inhibt']
    y_is_dead = is_dead.loc[:, '1 month Death']
    X_is_dead_train, X_is_dead_test, y_is_dead_train, y_is_dead_test = train_test_split(
        X_is_dead, y_is_dead, test_size=percent, random_state=42)
    y_is_dead_test = pd.concat([X_is_dead_test, y_is_dead_test], axis=1, sort=False)
    dataset = dataset.drop(index=y_is_dead_test.index.tolist())
    return dataset, y_is_dead_test

def extract_test(dataset,percent_dead,percent_alive) :
    dataset,dataset_dead=extract_label(dataset, 1,percent_dead)
    dataset,dataset_alive=extract_label(dataset, 0,percent_alive)
    return dataset,pd.concat([dataset_alive,dataset_dead])



def rebalance(dataset):
    X = dataset.loc[:, 'CenterID':'P2Y12 inhibt']
    y = dataset.loc[:, '1 month Death']
    dataset, X, y  = my_l_rebalance(X, y)
    return X, y, dataset

