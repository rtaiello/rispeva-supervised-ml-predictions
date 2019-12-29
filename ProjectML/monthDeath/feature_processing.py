# std lib

# my lib

# thid part lib
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE

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

