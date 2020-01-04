# std lib import

# my lib

# third part import
import pandas as pd
import numpy as np
from fancyimpute import KNN
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split

SEED = 1

def my_l_extract_feature(filename,label):
    dt = my_l_read(filename)
    X_month_death = dt.loc[:, 'CenterID':'P2Y12 inhibt']
    y_month_death = dt.loc[:, label]

    return pd.concat([X_month_death, y_month_death], axis=1, sort=False)

def my_l_read(filename):
    dataset = pd.read_excel(filename)
    return dataset

def my_l_imp(types,dataset):
    if(types == "KNN"):
        knn_imputer = KNN()
        dt_knn = dataset.copy(deep=True)
        dt_knn.iloc[:, :] = knn_imputer.fit_transform(dataset)
        return dt_knn, knn_imputer


def my_l_rebalance(X,y, percent):
    #rus = RandomUnderSampler(sampling_strategy=0.06, random_state=42)
    #X_res, y_res = rus.fit_resample(X, y)
    X_resampled, y_resampled = SMOTE(sampling_strategy=percent, random_state=42, k_neighbors=5).fit_resample(X, y)
    print("Label 1:")
    print(y_resampled[y_resampled==1].shape)
    print("Label 0:")
    print(y_resampled[y_resampled == 0].shape)
    dataset =  pd.concat([X_resampled, y_resampled], axis=1, sort=False)
    return dataset,X_resampled,y_resampled

def my_l_confusion_matrix(y_test,y_pred):
    cnf_matrix_dt = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cnf_matrix_dt.ravel()
    return (tn, fp, fn, tp)

def my_l_f1_scores(clf,X,y):

    return cross_val_score(clf, X, y, cv=5, scoring='f1')

def my_l_split(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X,y.astype('int'), stratify=y,random_state=SEED)
    return X_train, X_test, y_train, y_test

def imputation(dataset, label_name):
    row_removed = dataset[dataset[label_name].isnull()].index.tolist()
    dataset = dataset.drop(index=row_removed)
    d_month_mean_null = dataset.isnull().mean() * 100
    col_removed = d_month_mean_null[d_month_mean_null >= 60.0].index.tolist()
    dataset = dataset.drop(columns=col_removed)
    binary_cols = [col for col in dataset if np.isin(dataset[col].dropna().unique(), [0.0, 1.0]).all()]
    dataset = my_l_imp("KNN", dataset)[0]
    print(dataset[dataset.isna().any(axis=1)])
    dataset[binary_cols] = dataset[binary_cols].round()
    dataset.to_excel('../dataset/imputed_KNN.xlsx')
    return dataset, row_removed