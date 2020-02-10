# Standard lib

# Our import
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LassoCV
from sklearn.metrics import precision_score, recall_score, f1_score

from ProjectML.general_util import *
from ProjectML.monthDeath.pre_processing.pre_processing import *
from ProjectML.monthDeath.classification import *
from ProjectML.monthDeath.pre_processing.feature_processing import *
from ProjectML.evaluation import *


# Third part lib
from sklearn import svm
from sklearn.preprocessing import Normalizer
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt

# Constant
DONE_imputation = False
DATASET_FILENAME = '../dataset/RISPEVA_dataset_for_ML.xlsx'
DATASET_IMPUTATION = "../dataset/imputation_1MD.xlsx"
LABEL = '1 month Death'
dataset = None
# ---------- init imputation ----------
if not DONE_imputation:
    dataset = my_l_read(DATASET_FILENAME)
    dataset = imputation(dataset)
    dataset.to_excel(DATASET_IMPUTATION)
    print("IMPUTATION DONE!")

else:
    dataset = my_l_read(DATASET_IMPUTATION)
# ---------- end imputation ----------

# ---------- pick features ----------
X, y ,dataset = my_l_extract_feature(dataset, label=LABEL)

# ---------- init split test ----------
X, X_test, y, y_test = my_l_split(X,y,split_percent=0.1)
# ---------- end split validation ----------

X_train, X_val, y_train, y_val = my_l_split(X, y,split_percent=0.2)

print("Percent of death in original dataset= {0:.2f}".format(y[y==1].count()/y.count()))
print("Percent of death in test set= {0:.2f}".format(y_test[y_test==1].count()/y_test.count()))
print("Percent of death in validation set= {0:.2f}".format(y_val[y_val==1].count()/y_val.count()))

#get continuos features, with variance grater than 1

X_scaled_train, X_scaled_val = my_l_std_scaling(X_train, X_val)

X_res, y_res = my_l_rebalance(X_scaled_train,y_train,0.5)

clf = ensemble_ada_boosting(X_res,y_res)

y_pred = clf.predict(X_val)
print(report(y_val,y_pred))
# ---------- init f. importance ----------

# ---------- end f. importance ----------


