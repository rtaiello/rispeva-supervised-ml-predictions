# third part lib
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn.metrics import f1_score
import numpy as np
SEED=1


def my_l_confusion_matrix(y_test, y_pred):
    cnf_matrix_dt = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cnf_matrix_dt.ravel()
    return (tn, fp, fn, tp)


def evaluation(y_test, y_pred):
    return my_l_confusion_matrix(y_test, y_pred)


def get_f1_scores(X, y, clf):
    f1_scores = cross_val_score(clf, X, y, cv=5, scoring='f1')
    return f1_scores


def get_balanced_accuracy(X, y, clf):
    balanced_accuracy = cross_val_score(clf, X, y, cv=5, scoring='balanced_accuracy')
    return balanced_accuracy


def report(y_true, y_pred):
    class_report = classification_report(y_true, y_pred)
    return class_report

def my_cross_f1(X,y,clf,cv) :
    skf = StratifiedKFold(n_splits=cv, random_state=SEED, shuffle=True)
    f1_list_0 = []
    f1_list_1 = []
    train_list = []
    test_list = []
    for train_index, test_index in skf.split(X, y):
        train_list = pd.concat([X.iloc[train_index.tolist(), :], y.iloc[train_index.tolist()]], axis=1)
        test_list = pd.concat([X.iloc[test_index.tolist(), :], y.iloc[test_index.tolist()]], axis=1)
        clf.fit(X.iloc[train_index.tolist(), :], y.iloc[train_index.tolist()])
        y_predicted = clf.predict(X.iloc[test_index.tolist(), :])
        f1_list_0.append(f1_score(abs(y.iloc[test_index.tolist()] - 1), abs(y_predicted - 1)))
        f1_list_1.append(f1_score(y.iloc[test_index.tolist()], y_predicted))
    f1_list_0 = np.array(f1_list_0)
    f1_list_1 = np.array(f1_list_1)
    f1_mean0 = f1_list_0.mean()
    f1_std0 = f1_list_0.std()
    f1_mean1 = f1_list_1.mean()
    f1_std1 = f1_list_1.std()
    return f1_mean0, f1_std0, f1_mean1, f1_std1

