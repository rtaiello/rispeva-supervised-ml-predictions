# os library
import re
import os

# third part lib
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot  as plt
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

def my_l_plot_feature(feature, label_ax, path= None):

    def ecdf(data):
        import numpy as np
        # Empirical Cumulative Distribution Function
        """Compute ECDF for a one-dimensional array of measurements."""
        # Number of data points: n
        n = len(data)
        # x-data for the ECDF: x
        x = np.sort(data)
        # y-data for the ECDF: y
        y = np.arange(1, n + 1) / n
        return x, y

    x_amt_credit, y_amt_credit = ecdf(feature)

    # Generate plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('ECDF vs histogram:' + label_ax)
    ax1.plot(x_amt_credit, y_amt_credit, marker='.', linestyle='none', color='red')
    ax1.set(xlabel=label_ax, ylabel='ECDF')

    feature.hist(ax=ax2)
    ax2.set(xlabel=label_ax, ylabel='Frequency')

    # Display the plot
    if path:
        plt.savefig(path)
    else:
        plt.show()