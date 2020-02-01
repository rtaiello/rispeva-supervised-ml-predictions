# Standard lib

# Our import
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import precision_score, recall_score, f1_score

from ProjectML.general_util import *
from ProjectML.monthDeath.pre_processing.pre_processing import *
from ProjectML.monthDeath.classification import *
from ProjectML.monthDeath.pre_processing.feature_processing import *
from ProjectML.evaluation import *


# Third part lib
from sklearn import svm

# Constant
DONE_imputation = True
DATASET_FILENAME = '../dataset/RISPEVA_dataset.xlsx'
DATASET_IMPUTATION = "../dataset/imputation_1MD.xlsx"
LABEL = '1 month Death'

dataset = None
# ---------- init imputation ----------
if not DONE_imputation:
    dataset = my_l_read(DATASET_FILENAME)
    dataset, row_removed = imputation(dataset, label=LABEL)
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


#DA CUI CREARE UNA BASELINE CON SVM


# ---------- init Classifiers ----------
svm = svm_classifier(X_train, y_train)
random_forest = ensemble_random_forest(X_train, y_train)
ada_boost = ensemble_ada_boosting(X_train, y_train)
# ---------- end Classifiers ----------
#
# ---------- init SVM ----------
y_pred_svm=svm.predict(X_test)
# ---------- end SVM ----------
#
# ---------- init Random Forest ----------
y_pred_random_forest=random_forest.predict(X_test)
# ---------- end Random Forest ----------
#
# ---------- init Boosting ----------
#y_pred_boosting=ada_boost.predict(X_val)
# ---------- end Boosting ----------

print("SVM's prediction:")
print(report(y_test, y_pred_svm))

print("Random Forest's prediction")
print(report(y_test, y_pred_random_forest))

print("Ada Boost's prediction")
print(report(y_val, y_pred_boosting))

print("MLP's prediction")
print(report(y_test, y_pred_mlp))