# std lib

#  import foo

from ProjectML.general_util import *
from ProjectML.proceduralSuccess.pre_processing import *
from ProjectML.proceduralSuccess.evaluation import *
from ProjectML.proceduralSuccess.classification import *
from ProjectML.proceduralSuccess.feature_processing import *

DT_FILENAME = "../dataset/RISPEVA_dataset_for_ML.xlsx"
DT_IMPUTATION = "../dataset/imputed_KNN.xlsx"

dt = my_l_extract_feature(DT_IMPUTATION, 'ProceduralSuccess')
#print(dt.head())
#dt = imputation(dt, 'ProceduralSuccess')[0]


#dt, dt_test = extract_test(dt, 'ProceduralSuccess', 0.10, 0.20)
#X_test = dt_test.loc[:, 'CenterID':'P2Y12 inhibt']
#y_test = dt_test.loc[:, 'ProceduralSuccess']

dt_X = dt.drop(['ProceduralSuccess'],axis=1)
dt_y = dt[['ProceduralSuccess']]

X_train, X_test, y_train, y_test = train_test_split(dt_X,dt_y,stratify=dt_y)

# X, y, dt = rebalance(dt, 'ProceduralSuccess')
result = feature_importance(X, y, 30)
result = (list(list(zip(*result))[0]))
X = X.loc[:, result]

# # result=feature_selection(X,y)
# # X = X.loc[:, result]

clf = ensemble_stacking(X, y)
clf2 = ensemble_random_forest(X, y)
y_pred_stack = clf.predict(X_test.loc[:, result])
y_pred_random = clf2.predict(X_test.loc[:, result])

print(report(y_pred_stack, y_test))

print(report(y_pred_random, y_test))
