# third part lib
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report


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
