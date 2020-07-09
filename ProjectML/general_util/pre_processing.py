# os library
import re
import os

# third part lib
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.model_selection import train_test_split
import pandas as pd

# constant
from ProjectML.general_util.constant import *


def my_l_read_dataframe(filename: str):
    print(os.getcwd())
    dataset = pd.read_csv(filename) if filename.endswith(".csv") else pd.read_excel(filename)
    my_l_rm_white_space(dataset)
    return dataset


def my_l_rm_white_space(dataset):
    regex = re.compile(r"[\[\]<]", re.IGNORECASE)
    dataset.columns = [regex.sub("_", col) if any(x in str(col) for x in {'[', ']', '<'}) else col for col in
                       dataset.columns.values]
    dataset.columns = dataset.columns.str.replace(' ', '')
    return dataset


def my_l_std_scaling(X):
    feature_selected = X.select_dtypes(include=['int64', 'float64'])
    X_scaled = X.copy()
    X_scaled[feature_selected.columns] = StandardScaler().fit_transform(X_scaled[feature_selected.columns])
    return X_scaled


def my_l_norm_scaling(X):
    feature_selected = X.select_dtypes(include=['int64', 'float64'])
    X_scaled = X.copy()
    X_scaled[feature_selected.columns] = Normalizer().fit_transform(X_scaled[feature_selected.columns])
    return X_scaled


def my_l_split(X, y, split_percent):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_percent, random_state=SEED,
                                                        stratify=y, shuffle=True)
    return X_train, X_test, y_train, y_test
