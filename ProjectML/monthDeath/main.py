# Standard lib

# Our import
from ProjectML.general_util import *
from ProjectML.monthDeath.pre_processing.pre_processing import *
from ProjectML.monthDeath.classification import *
from ProjectML.monthDeath.pre_processing.feature_processing import *
from ProjectML.evaluation import *

# Third part lib

# Constant
DONE_imputation = False
DATASET_FILENAME = 'dataset/RISPEVA_dataset.xlsx'
DATASET_IMPUTATION = "dataset/imputation_1MD.xlsx"
LABEL = '1 moth Death'

dataset = None
# ---------- init imputation ----------
if not DONE_imputation:
    dataset = my_l_read(DATASET_FILENAME)
    dataset, row_removed = imputation(dataset)
    dataset.to_excel(DATASET_IMPUTATION)
    print("IMPUTATION DONE!")
    print("Row removed:"+row_removed)
else:
    dataset = my_l_read(DATASET_IMPUTATION)
# ---------- end imputation ----------

# ---------- pick features ----------
dataset = my_l_extract_feature(dataset, label=LABEL)

# ---------- init split validation ----------
dataset, dataset_validation = extract_test(dataset,percent_dead= 0.22, percent_alive= 0.03)
dt_test_X=dataset_validation.loc[:,'CenterID':'P2Y12 inhibt']
dt_test_y=dataset_validation.loc[:, LABEL]
# ---------- end split validation ----------

# ---------- init rebalance ----------
X, y, dataset = rebalance(dataset, percent=0.12)
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
y_pred_random_forest=random_forest.predict(X_train)

print(report(y_pred_random_forest,y_test))


