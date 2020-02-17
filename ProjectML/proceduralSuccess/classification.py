#std lib

#my lib

#third part lib
import numpy as np
from sklearn import svm
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, VotingClassifier, RandomForestRegressor, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans

SEED = 1

def kmeans_clustering(X_train, k):
    clf = KMeans(n_clusters=k, random_state=SEED).fit(X_train)
    return clf

def mlp_classifier(X_train, y_train):
    clf = MLPClassifier(hidden_layer_sizes=(100,100,100),max_iter=500)
    clf.fit(X_train, y_train)
    return clf

def decision_tree_classifier(X_train, y_train,classWeights):
    clf = DecisionTreeClassifier(criterion='gini',random_state=SEED, max_features=40, class_weight=classWeights)
    clf.fit(X_train, y_train)
    return clf


def ensemble_bagging(X_train, y_train):
    classif = svm.SVC(kernel='linear', C=1.0, max_iter=-1,  random_state=SEED)
    clf = BaggingClassifier(base_estimator=classif, n_estimators=30, n_jobs=-1, random_state=SEED)
    clf.fit(X_train, y_train)
    return clf

def ensemble_random_forest(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=500, criterion='gini', max_features=5, n_jobs=-1, class_weight='balanced', random_state=SEED)
    # Fit 'rf' to the training set
    clf.fit(X_train, y_train)
    # Predict the test set labels 'y_pred'
    return clf

def ensemble_ada_boosting(X_train, y_train):
    classifier = svm.SVC(C=1, kernel='linear', class_weight='balanced', max_iter=-1, random_state=SEED)
    #classifier = DecisionTreeClassifier(random_state=SEED)
    clf = AdaBoostClassifier(base_estimator=classifier, n_estimators=50, algorithm='SAMME')
    clf.fit(X_train, y_train)
    return clf

def svm_classifier(X_train, y_train):
    #x_scaled = StandardScaler().fit_transform(X_train)
    clf = svm.SVC(C=100.0, kernel='rbf', class_weight='balanced', max_iter=-1,  random_state=SEED).fit(X_train, y_train)
    return clf

def ensemble_voting(X_train,y_train):
    classifiers = [('Random Forest', RandomForestClassifier(n_estimators=300,
                                                            min_samples_leaf=0.12,
                                                            random_state=SEED)),
                   ('SVM Classifier', svm.SVC(probability=True))]
    clf = VotingClassifier(estimators=classifiers,voting='soft',n_jobs=-1,weights=[1.5, 2])
    clf.fit(X_train, y_train)
    return clf

def ensemble_stacking(X_train, y_train):
    estimators = [('dt', DecisionTreeClassifier()),('svm',
                    make_pipeline(StandardScaler(), svm.SVC(random_state=42)))]
    clf = StackingClassifier(estimators = estimators, final_estimator = DecisionTreeClassifier())
    clf.fit(X_train, y_train)
    return clf

def my_voting(clf1,clf2,X_test) :
    y_pred_1 = clf1.predict(X_test)
    y_pred_2 = clf2.predict(X_test)
    return np.logical_or (y_pred_1, y_pred_2)
