# Standard lib
import os
from os import sep

# Our import
import pandas as pd
import numpy as np
from fancyimpute import KNN
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split

# Constant
SEED = 1


def my_l_extract_feature(dataset, label):
    X_month_death = dataset.loc[:, 'CenterID':'P2Y12 inhibt']
    y_month_death = dataset.loc[:, label]

    return X_month_death, y_month_death, pd.concat([X_month_death, y_month_death], axis=1, sort=False)


def my_l_read(filename):
    print(os.getcwd())
    dataset = pd.read_excel(filename)
    return dataset


def my_l_imp_KNN(dataset):
    knn_imputer = KNN()
    dt_knn = dataset.copy(deep=True)
    dt_knn.iloc[:, :] = knn_imputer.fit_transform(dataset)
    return dt_knn, knn_imputer


def my_l_rebalance(X, y, percent):
    X_resampled, y_resampled = SMOTE(sampling_strategy=percent, random_state=42, k_neighbors=3).fit_resample(X, y)
    dataset = pd.concat([X_resampled, y_resampled], axis=1, sort=False)
    return dataset, X_resampled, y_resampled


def my_l_confusion_matrix(y_test, y_pred):
    cnf_matrix_dt = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cnf_matrix_dt.ravel()
    return (tn, fp, fn, tp)


def my_l_f1_scores(clf, X, y):
    return cross_val_score(clf, X, y, cv=5, scoring='f1')


def my_l_split(X, y, split_percent):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_percent, random_state=SEED,
                                                        stratify=y,shuffle=True)
    return X_train, X_test, y_train, y_test


def imputation(dataset, label):
    row_removed = dataset[dataset[label].isnull()].index.tolist()
    dataset = dataset.drop(index=row_removed)
    d_month_mean_null = dataset.isnull().mean() * 100
    col_removed = d_month_mean_null[d_month_mean_null >= 60.0].index.tolist()
    dataset = dataset.drop(columns=col_removed)
    binary_cols = [col for col in dataset if np.isin(dataset[col].dropna().unique(), [0.0, 1.0]).all()]
    dataset = my_l_imp_KNN(dataset)[0]
    print(dataset[dataset.isna().any(axis=1)])
    dataset[binary_cols] = dataset[binary_cols].round()
    return dataset, row_removed
