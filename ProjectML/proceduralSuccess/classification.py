#std lib

#my lib

#third part lib
import numpy as np
from sklearn import svm
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, VotingClassifier, \
    RandomForestRegressor, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

SEED = 1

def decision_tree_classifier(X_train, y_train):
    clf = DecisionTreeClassifier(random_state=SEED)
    clf.fit(X_train, y_train)
    return clf

def ensemble_bagging(X_train, y_train):
    classifier = RandomForestClassifier(n_estimators=100,random_state=SEED,
                                n_jobs=-1)
    clf = BaggingClassifier(base_estimator=classifier, n_estimators=100, n_jobs=-1)
    clf.fit(X_train, y_train)
    return clf

def ensemble_random_forest(X_train, y_train):

    clf = RandomForestClassifier(n_estimators=300, random_state=SEED)
    clf.fit(X_train, y_train)
    return clf

def ensemble_ada_boosting(X_train, y_train):
    clf = AdaBoostClassifier(n_estimators=200, algorithm='SAMME')
    clf.fit(X_train, y_train)
    return clf

def svm_classifier(X_train,y_train):
    X_scaled = StandardScaler().fit_transform(X_train)
    clf = svm.SVC().fit(X_scaled, y_train)
    return clf

def ensemble_voting(X_train,y_train):
    classifiers = [('Random Forest', RandomForestClassifier(n_estimators=300,
                                                            min_samples_leaf=0.12,
                                                            random_state=SEED)),
                   ('SVM Classifier', svm.SVC(probability=True))]
    clf = VotingClassifier(estimators=classifiers,voting='soft',n_jobs=-1)
    clf.fit(X_train, y_train)
    return clf

def ensemble_stacking(X_train, y_train):
    estimators = [('rf', RandomForestClassifier(n_estimators=300, min_samples_leaf=0.12, random_state=42)),('svr',
                    make_pipeline(StandardScaler(), svm.SVC(random_state=42)))]
    clf = StackingClassifier( estimators = estimators, final_estimator = svm.SVC())
    clf.fit(X_train, y_train)
    return clf
