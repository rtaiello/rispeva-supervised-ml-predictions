# std lib

#  import foo

from ProjectML.general_util import *
from ProjectML.proceduralSuccess.pre_processing import *
from ProjectML.proceduralSuccess.evaluation import *
from ProjectML.proceduralSuccess.classification import *
from ProjectML.proceduralSuccess.feature_processing import *

DT_FILENAME = "../dataset/RISPEVA_dataset_for_ML.xlsx"
DT_KNN = "../dataset/KNN.xlsx"

dt = my_l_extract_feature(DT_KNN, 'ProceduralSuccess')
# dt = imputation(dt, 'ProceduralSuccess')[0]


dt, dt_test = extract_test(dt, 'ProceduralSuccess', 0.20, 0.10)
dt_test_X = dt_test.loc[:, 'CenterID':'P2Y12 inhibt']
dt_test_y = dt_test.loc[:, 'ProceduralSuccess']

X, y, dt = rebalance(dt, 'ProceduralSuccess')
result = feature_importance(X, y, 30)
result = (list(list(zip(*result))[0]))
X = X.loc[:, result]

# result=feature_selection(X,y)
# X = X.loc[:, result]


X_train, X_test, y_train, y_test = my_l_split(X, y)
clf = ensemble_stacking(X_train, y_train)
clf2 = ensemble_random_forest(X_train, y_train)
y_pred_stack = clf.predict(dt_test_X.loc[:, result])
y_pred_random = clf2.predict(dt_test_X.loc[:, result])

print(report(y_pred_stack, dt_test_y))

print(report(y_pred_random, dt_test_y))
