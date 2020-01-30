# Standard lib

# Our import
from ProjectML.general_util import *
from ProjectML.monthDeath.pre_processing.pre_processing import *
from ProjectML.monthDeath.classification import *
from ProjectML.monthDeath.pre_processing.feature_processing import *
from ProjectML.evaluation import *

# Third part lib

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
dataset = my_l_extract_feature(dataset, label=LABEL)

# ---------- init split validation ----------
dataset, dataset_validation = extract_test(dataset,percent_dead= 0.22, percent_alive= 0.03)
X_validation = dataset_validation.loc[:,'CenterID':'P2Y12 inhibt']
y_validation = dataset_validation.loc[:, LABEL]
# ---------- end split validation ----------

# ---------- init rebalance ----------
X, y, dataset = rebalance(dataset, percent=1)
# ---------- end rebalance ----------

# ---------- init f. importance ----------
result = feature_importance(X,y,25)
result = (list(list(zip(*result))[0]))
X = X.loc[:, result]
# ---------- end f. importance ----------

X_train, X_test, y_train, y_test = my_l_split(X, y)
svm = svm_classifier(X, y)
random_forest = ensemble_random_forest(X, y)

y_pred_svm=svm.predict(X_test)
y_pred_random_forest=random_forest.predict(X_test)

print("SVM's prediction:")
print(report(y_test, y_pred_svm))

print("Random Forest's prediction")
print(report(y_test, y_pred_random_forest))

