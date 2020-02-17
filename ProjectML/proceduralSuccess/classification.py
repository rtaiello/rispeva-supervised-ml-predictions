# std lib

# my lib

# third part lib
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

# constant
from ProjectML.general_util.constant import SEED


def ensemble_random_forest(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=500, criterion='gini', max_features=5, n_jobs=-1, class_weight='balanced',
                                 random_state=SEED)
    # Fit 'rf' to the training set
    clf.fit(X_train, y_train)
    return clf


def svm_classifier(X_train, y_train):
    clf = svm.SVC(C=100.0, kernel='rbf', class_weight='balanced', max_iter=-1, random_state=SEED).fit(X_train, y_train)
    return clf
