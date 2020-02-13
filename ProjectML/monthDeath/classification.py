# std lib

# my lib

# third part lib
import numpy as np
from sklearn import svm
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, VotingClassifier, \
    StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

SEED = 42


def decision_tree_classifier(X_train, y_train):
    clf = DecisionTreeClassifier(random_state=SEED)
    clf.fit(X_train, y_train)
    return clf


def ensemble_bagging(X_train, y_train):
    classifier = AdaBoostClassifier(LogisticRegression(class_weight='balanced', n_jobs=-1), n_estimators=50,
                                    learning_rate=1.0, algorithm='SAMME')
    clf = BaggingClassifier(base_estimator=classifier, n_estimators=100, n_jobs=-1)
    clf.fit(X_train, y_train)
    return clf


def ensemble_random_forest(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=100,
                                 min_samples_leaf=0.12,
                                 random_state=SEED)
    # Fit 'rf' to the training set
    clf.fit(X_train, y_train)
    # Predict the test set labels 'y_pred'
    return clf


def ensemble_ada_boosting(X_train, y_train):
    clf = AdaBoostClassifier(LogisticRegression(class_weight='balanced', max_iter=1000, random_state=SEED, n_jobs=-1), n_estimators=100, learning_rate=1.0,
                             algorithm='SAMME')
    clf.fit(X_train, y_train)
    return clf


def svm_classifier(X_train, y_train):
    clf = svm.SVC(kernel='rbf',class_weight='balanced',random_state=SEED).fit(X_train, y_train)
    return clf


def ensemble_voting(X_train, y_train):
    clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
    clf3 = GaussianNB()

    eclf1 = VotingClassifier(estimators=[
        ('rf', clf2), ('gnb', clf3)], voting='soft', weights=[2, 1], flatten_transform=True)
    eclf1.fit(X_train, y_train)
    return eclf1


def ensemble_stacking(X_train, y_train):
    estimators = [('rf', RandomForestClassifier(n_estimators=300, min_samples_leaf=0.12, random_state=42)), ('svr',
                                                                                                             make_pipeline(
                                                                                                                 StandardScaler(),
                                                                                                                 svm.SVC(
                                                                                                                     random_state=42)))]
    clf = StackingClassifier(estimators=estimators, final_estimator=DecisionTreeClassifier())
    clf.fit(X_train, y_train)
    return clf


def xgb_classifier(X_train, y_train, percent_minority):
    clf = XGBClassifier(scale_pos_weight=percent_minority, n_jobs=-1, random_state=SEED)
    clf.fit(X_train, y_train)
    return clf
