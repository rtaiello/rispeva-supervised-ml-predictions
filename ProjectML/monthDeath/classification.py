# my lib
from ProjectML.general_util.constant import *
# third part lib
from xgboost import XGBClassifier


def xgb_classifier(X_train, y_train):
    clf = XGBClassifier(colsample_bytree=1.0, gamma=5, max_delta_step=1, max_depth=7, min_child_weight=9,
                        scale_pos_weight=5, subsample=1.0, n_jobs=-1, random_state=SEED, n_estimators=100)
    clf.fit(X_train, y_train)
    return clf
