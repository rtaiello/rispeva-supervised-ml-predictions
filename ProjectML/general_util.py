# Standard lib
import os
from os import sep

# Our import
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

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

def my_l_std_scaling(X_train, X_test):
    feature_selected = X_train.select_dtypes(include=['int64','float64'])
    X_train[feature_selected.columns] = StandardScaler().fit_transform(X_train[feature_selected.columns])
    X_test[feature_selected.columns] = StandardScaler().fit_transform(X_test[feature_selected.columns])
    return X_train,X_test

def my_l_imp_KNN(dataset):
    KNN_imputer = KNNImputer()
    dt_knn = dataset.copy(deep=True)
    dt_knn.iloc[:, :] = KNN_imputer.fit_transform(dataset)
    return dt_knn


def my_l_rebalance(X, y, percent):
    X_resampled, y_resampled = SMOTE(sampling_strategy=percent, random_state=42, k_neighbors=5).fit_resample(X, y)
    return X_resampled, y_resampled


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


