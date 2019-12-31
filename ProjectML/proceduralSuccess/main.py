# std lib

#

from ProjectML.general_util import *
from ProjectML.proceduralSuccess.pre_processing import *
from ProjectML.proceduralSuccess.evaluation import *
from ProjectML.proceduralSuccess.classification import *
from ProjectML.proceduralSuccess.feature_processing import *

DT_FILENAME = "../dataset/RISPEVA_dataset_for_ML.xlsx"
DT_KNN = "../dataset/imputed_KNN.xlsx"
LABEL = 'ProceduralSuccess'

dt = my_l_extract_feature(DT_KNN, LABEL)
#dt = imputation(dt, 'ProceduralSuccess')[0]
X = dt.loc[:, 'CenterID':'P2Y12 inhibt']
y = y = dt.loc[:, LABEL]

mask = feature_variance(X)  #imposing threshold to features' variance
X = X.loc[:, mask]

X_train, X_test, y_train, y_test = my_l_split(X, y)
mask = voting_feature_selection(X_train, y_train)
X_train = X_train.loc[:, mask]
X_test = X_test.loc[:, mask]

clf = ensemble_random_forest(X_train, y_train)
#clf= svm_classifier(X_train, y_train)
y_pred_stack = clf.predict(X_test)
print(report(y_pred_stack, y_test))



