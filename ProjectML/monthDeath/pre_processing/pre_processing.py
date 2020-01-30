# std lib

# my lib
from ProjectML.general_util import *
# third part
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Constant
LABEL = '1 month Death'

def extract_label(dataset, label, percent):
    is_dead = dataset[dataset[LABEL] == label]
    X_is_dead = is_dead.loc[:, 'CenterID':'P2Y12 inhibt']
    y_is_dead = is_dead.loc[:, LABEL]
    X_is_dead_train, X_is_dead_test, y_is_dead_train, y_is_dead_test = train_test_split(
        X_is_dead, y_is_dead, test_size=percent, random_state=42)
    y_is_dead_test = pd.concat([X_is_dead_test, y_is_dead_test], axis=1, sort=False)
    dataset = dataset.drop(index=y_is_dead_test.index.tolist())
    return dataset, y_is_dead_test


def extract_test(dataset, percent_dead, percent_alive):
    dataset, dataset_dead = extract_label(dataset, 1, percent_dead)
    dataset, dataset_alive = extract_label(dataset, 0, percent_alive)
    return dataset, pd.concat([dataset_alive, dataset_dead])


def rebalance(dataset, percent):
    X = dataset.loc[:, 'CenterID':'P2Y12 inhibt']
    y = dataset.loc[:, LABEL]
    dataset, X, y = my_l_rebalance(X, y, percent)
    return X, y, dataset
