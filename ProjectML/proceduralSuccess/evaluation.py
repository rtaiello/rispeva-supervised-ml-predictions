# std lib

from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

# my lib
# third part lib
from ProjectML.general_util import my_l_confusion_matrix


def evaluation(y_test, y_pred):
    return my_l_confusion_matrix(y_test, y_pred)


def get_f1_scores(X, y, clf):
    f1_scores = cross_val_score(clf, X, y, cv=5, scoring='f1')
    return f1_scores


def report(y_true, y_pred):
    class_report = classification_report(y_true, y_pred)
    return class_report
