# my lib

from ProjectML.monthDeath.pre_processing import *
from ProjectML.monthDeath.feature_extraction import *
from ProjectML.general_util.pre_processing import *
from ProjectML.monthDeath.classification import *
from ProjectML.general_util.evaluation import *

# Constant
DONE_imputation = True
DATASET_FILENAME = '../../dataset/RISPEVA_dataset_for_ML.xlsx'
DATASET_IMPUTATION = "../../pickle/1MD_imputation.pkl"
LABEL = '1monthDeath'
dataset = None
# ---------- init imputation ----------
if not DONE_imputation:
    dataset = read_dataset(DATASET_FILENAME)
    dataset = imputation(dataset)
    dataset.to_pickle(DATASET_IMPUTATION)
    print("IMPUTATION DONE!")

else:
    dataset = pd.read_pickle(DATASET_IMPUTATION)
# ---------- end imputation ----------

# ----------  init features selection----------
X, y, dataset = extract_feature(dataset)
X = drop_corr_feature(X, 0.6)
X = best_eight_features(X)
# ----------  end features selection----------

# ---------- init split test ----------
X, X_test, y, y_test = my_l_split(X, y, split_percent=0.1)
# ---------- end split test ----------

# ---------- init split train validation ----------
X_train, X_val, y_train, y_val = my_l_split(X, y, split_percent=0.2)
# ---------- end split train validation ----------

print("Percent of death in original dataset= {0:.2f}".format(y[y == 1].count() / y.count()))
print("Percent of death in test set= {0:.2f}".format(y_test[y_test == 1].count() / y_test.count()))
print("Percent of death in validation set= {0:.2f}".format(y_val[y_val == 1].count() / y_val.count()))

print("-DROPPED FEATURES WITH CORR GRATER THAN 60 - FEATURES IMPORTANCE: THE BEST 8 - IMBALANCED DATASET-")
xgb = xgb_classifier(X_train, y_train)
y_pred = xgb.predict(X_val)
print(report(y_val, y_pred))
print("Cross validation cv = 5 ")
balanced_accuracy_score = get_balanced_accuracy(X, y, xgb)
print("Balanced accuracy: %0.2f (+/- %0.2f)" % (balanced_accuracy_score.mean(), balanced_accuracy_score.std() * 2))
f1_score = get_f1_scores(X, y, xgb)
print("f1_score: %0.2f (+/- %0.2f)" % (f1_score.mean(), f1_score.std() * 2))