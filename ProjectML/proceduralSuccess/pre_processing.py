# std lib

# my lib
from ProjectML.general_util import my_l_imp, my_l_rebalance
# third part
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np





def extract_label(dataset,label_name,label,percent):
    is_dead = dataset[dataset[label_name] == label]
    X_is_dead = is_dead.loc[:, 'CenterID':'P2Y12 inhibt']
    y_is_dead = is_dead.loc[:, label_name]
    X_is_dead_train, X_is_dead_test, y_is_dead_train, y_is_dead_test = train_test_split(
        X_is_dead, y_is_dead, test_size=percent, random_state=42)
    y_is_dead_test = pd.concat([X_is_dead_test, y_is_dead_test], axis=1, sort=False)
    dataset = dataset.drop(index=y_is_dead_test.index.tolist())
    return dataset, y_is_dead_test

def extract_test(dataset,label_name, percent_1,percent_0) :
    dataset, dataset_dead = extract_label(dataset, label_name, 1, percent_1)
    dataset, dataset_alive = extract_label(dataset,label_name, 0, percent_0)
    return dataset, pd.concat([dataset_alive,dataset_dead])


def rebalance(dataset,label_name):
    X = dataset.loc[:, 'CenterID':'P2Y12 inhibt']
    y = dataset.loc[:, label_name]
    dataset, X, y  = my_l_rebalance(X, y, 0.7)
    return X, y, dataset

