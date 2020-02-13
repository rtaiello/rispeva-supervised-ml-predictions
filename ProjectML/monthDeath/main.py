# my lib

from ProjectML.monthDeath.pre_processing import *
from ProjectML.monthDeath.feature_extraction import *
from ProjectML.general_util.pre_processing import *
from ProjectML.monthDeath.classification import *
from ProjectML.general_util.evaluation import *
from ProjectML.general_util.rebalance import *

# Constant
DONE_imputation = True
DATASET_FILENAME = '../../dataset/RISPEVA_dataset_for_ML.xlsx'
DATASET_IMPUTATION = "../../pickle/1MD_imputation.pkl"
LABEL = '1monthDeath'
dataset = None
# ---------- init imputation ----------
if not DONE_imputation:
    dataset = my_l_read(DATASET_FILENAME)
    dataset = imputation(dataset)
    dataset.to_pickle(DATASET_IMPUTATION)
    print("IMPUTATION DONE!")

else:
    dataset = pd.read_pickle(DATASET_IMPUTATION)
# ---------- end imputation ----------

# ---------- pick features ----------
X, y, dataset = extract_feature(dataset, label=LABEL)
X = drop_corr_feature(X)
# ---------- init split test ----------
X, X_test, y, y_test = my_l_split(X, y, split_percent=0.1)
# ---------- end split validation ----------

X_train, X_val, y_train, y_val = my_l_split(X, y, split_percent=0.2)

print("Percent of death in original dataset= {0:.2f}".format(y[y == 1].count() / y.count()))
print("Percent of death in test set= {0:.2f}".format(y_test[y_test == 1].count() / y_test.count()))
print("Percent of death in validation set= {0:.2f}".format(y_val[y_val == 1].count() / y_val.count()))

# get continuos features, with variance grater than 1
print("DROPPED FEATURES WITH CORR GRATER THAN 80, IMBALANCED DATASET")
clf = xgb_classifier(X_train, y_train, 4)
y_pred = clf.predict(X_val)
print(report(y_val, y_pred))
print("Balanced accuracy")
print(get_balanced_accuracy(X, y, clf))
print("f1_scores")
print(get_f1_scores(X, y, clf))
print("#########################")
print("DROPPED FEATURES WITH CORR GRATER THAN 80, SMOTE 0.5 DATASET")
X_res, y_res = my_l_SMOTE_sampling(X_train, y_train, 0.5)
clf = xgb_classifier(X_res, y_res, 2)
y_pred = clf.predict(X_val)
print(report(y_val, y_pred))
print("Balanced accuracy")
print(get_balanced_accuracy(X, y, clf))
print("f1_scores")
print(get_f1_scores(X, y, clf))
print("DROPPED FEATURES WITH CORR GRATER THAN 80, ADASYN 0.5 DATASET")
X_res, y_res = my_l_ADASYN_sampling(X_train, y_train, 0.5)
clf = xgb_classifier(X_res, y_res, 2)
y_pred = clf.predict(X_val)
print(report(y_val, y_pred))
print("Balanced accuracy")
print(get_balanced_accuracy(X, y, clf))
print("f1_scores")
print(get_f1_scores(X, y, clf))
print("DROPPED FEATURES WITH CORR GRATER THAN 80, UNDERSAMPLING 0.5 DATASET")
X_res, y_res = my_l_under_sampling(X_train, y_train, 0.5)
clf = xgb_classifier(X_res, y_res, 2)
y_pred = clf.predict(X_val)
print(report(y_val, y_pred))
print("Balanced accuracy")
print(get_balanced_accuracy(X, y, clf))
print("f1_scores")
print(get_f1_scores(X, y, clf))

print("LOGISTIC REGRESSION: DROPPED FEATURES WITH CORR GRATER THAN 80, IMBALANCED DATASET")
X_train_scaled = my_l_norm_scaling(X_train)
X_val_scaled = my_l_norm_scaling(X_val)
clf = LogisticRegression(class_weight='balanced',penalty='l2',C=0.1, max_iter=1000, random_state=SEED, n_jobs=-1)
clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_val_scaled)
print(report(y_val, y_pred))
print("Balanced accuracy")
print(get_balanced_accuracy(my_l_std_scaling(X), y, clf))
print("f1_scores")
print(get_f1_scores(my_l_std_scaling(X), y, clf))

print("ADABOOST: DROPPED FEATURES WITH CORR GRATER THAN 80, IMBALANCED DATASET")
X_train_scaled = my_l_std_scaling(X_train)
X_val_scaled = my_l_std_scaling(X_val)
clf = ensemble_ada_boosting(X_train,y_train)
clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_val_scaled)
print(report(y_val, y_pred))
print("Balanced accuracy")
print(get_balanced_accuracy(my_l_std_scaling(X), y, clf))
print("f1_scores")
print(get_f1_scores(my_l_std_scaling(X), y, clf))

print("SVM: DROPPED FEATURES WITH CORR GRATER THAN 80, IMBALANCED DATASET")
X_train_scaled = my_l_std_scaling(X_train)
X_val_scaled = my_l_std_scaling(X_val)
clf = svm_classifier(X_train_scaled,y_train)
clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_val_scaled)
print(report(y_val, y_pred))
print("Balanced accuracy")
print(get_balanced_accuracy(my_l_std_scaling(X), y, clf))
print("f1_scores")
print(get_f1_scores(my_l_std_scaling(X), y, clf))