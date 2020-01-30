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

def feature_selection(X,y) :
    rfe = RFE(estimator=ExtraTreesClassifier(), n_features_to_select=40,step=10,verbose=1)
    rfe.fit(X,y)
    return rfe.support_

def voting_feature_selection(X_train,y_train, n_features):

    rfe_rf = RFE(estimator=RandomForestClassifier(),
                 n_features_to_select=n_features, step=5, verbose=1)
    rfe_rf.fit(X_train, y_train)
    rf_mask = rfe_rf.support_

    rfe_gb = RFE(estimator=GradientBoostingClassifier(),
                 n_features_to_select=n_features, step=5, verbose=1)
    rfe_gb.fit(X_train, y_train)
    gb_mask = rfe_gb.support_

    lcv = LassoCV()
    lcv.fit(X_train, y_train)

    lcv_mask = lcv.coef_ != 0

    votes = np.sum([lcv_mask, rf_mask, gb_mask], axis=0)

    mask = votes >=2
    return mask;


def feature_variance(X):
    sel = VarianceThreshold(threshold=0.005)
    sel.fit(X / X.mean())
    mask = sel.get_support()
    return  mask;