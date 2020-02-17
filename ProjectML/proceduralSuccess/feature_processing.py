# std lib

# my lib

# thid part lib
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import RFE, VarianceThreshold
from sklearn.linear_model import LassoCV
import numpy as np

SEED = 1


def feature_importance(X, y, n_features):
    model = ExtraTreesClassifier()
    # Fit 'rf' to the training set
    model.fit(X, y)
    result = list(zip(X.columns, model.feature_importances_))
    result.sort(key=lambda x: x[1], reverse=True)
    return result[:n_features]


def feature_selection(X, y):
    rfe = RFE(estimator=ExtraTreesClassifier(), n_features_to_select=40, step=10, verbose=1)
    rfe.fit(X, y)
    return rfe.support_


def voting_feature_selection(X_train, y_train):
    rfe_rf = RFE(estimator=RandomForestClassifier(), n_features_to_select=30, step=5, verbose=1)
    rfe_rf.fit(X_train, y_train)
    rf_mask = rfe_rf.support_
    rf_coefs = rfe_rf.estimator_.feature_importances_
    rfe_gb = RFE(estimator=GradientBoostingClassifier(),
                 n_features_to_select=30, step=5, verbose=1)
    rfe_gb.fit(X_train, y_train)
    gb_mask = rfe_gb.support_
    gb_coefs = rfe_gb.estimator_.feature_importances_
    # lcv = LassoCV()
    # lcv.fit(X_train, y_train)
    # lcv_mask = lcv.coef_ != 0
    votes = np.sum([rf_mask, gb_mask], axis=0)
    mask = votes >= 2
    return mask, gb_coefs, gb_mask, rf_coefs, rf_mask


def feature_variance(X):
    sel = VarianceThreshold(threshold=0.005)
    sel.fit(X / X.mean())
    mask = sel.get_support()
    return mask;


def cor_selector(X, y, num_feats):
    cor_list = []
    feature_name = X.columns.tolist()
    # calculate the correlation with y for each feature
    for i in X.columns.tolist():
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append(cor)
    # replace NaN with 0
    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    # feature name
    cor_feature = X.iloc[:, np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()
    # feature selection? 0 for not select, 1 for select
    cor_support = [True if i in cor_feature else False for i in feature_name]
    return cor_support, cor_feature, np.dot(np.array(cor_list), np.array(cor_support))
