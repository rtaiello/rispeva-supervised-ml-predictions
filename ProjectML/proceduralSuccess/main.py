# std lib

# my lib

from ProjectML.proceduralSuccess.pre_processing import imputation, read_dataset, extract_feature
from ProjectML.proceduralSuccess.classification import *
from ProjectML.proceduralSuccess.feature_extraction import *
from ProjectML.general_util.pre_processing import *
from ProjectML.general_util.evaluation import *
# third part
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import pandas as pd
# constant
from ProjectML.general_util.constant import *

DONE_IMPUTATION = True
DATASET_IMPUTATION = "../../pickle/PS/PS_imputation.pkl"
SCALED_FEATURES = "../../pickle/PS/PS_scaled_features.pkl"
LABEL = 'ProceduralSuccess'

# ---------- init imputation ----------
if not DONE_IMPUTATION:
    dataset = read_dataset(DATASET_FILENAME)
    dataset, missing_label = imputation(dataset)
    dataset.to_pickle(DATASET_IMPUTATION)
    print("IMPUTATION DONE!")

else:
    dataset = pd.read_pickle(DATASET_IMPUTATION)
# ---------- end imputation ----------

# ---------- init features selection ----------

X, y, dataset = extract_feature(dataset)
X_scaled = pd.read_pickle(SCALED_FEATURES)

# feature selection with Pearson correlation
# cor_support, cor_feature, cor_list = cor_selector(X, y, 21)
# print(str(len(cor_feature)), 'selected features')
# print(cor_feature)
# X = dt.loc[:, cor_feature]
# print(cor_list)

# feature selection with RFE
# clf = svm.SVC(C=1000.0, kernel='rbf', class_weight='balanced', max_iter=-1,  random_state=SEED)
# rfe_selector = RFE(estimator=clf, n_features_to_select=30, step=10, verbose=5)
# rfe_selector.fit(X_scaled, y)
# rfe_support = rfe_selector.get_support()
# rfe_feature = X.loc[:,rfe_support].columns.tolist()
# print(str(len(rfe_feature)), 'selected features')

mask, gb_coefs, gb_mask, rf_coefs, rf_mask = voting_feature_selection(X, y)
X = X.loc[:, mask]

X_scaled = X_scaled.loc[:, mask]
# ---------- end features selection ----------

# ---------- init split test ----------
X_train_val, X_test, y_train_val, y_test = my_l_split(X, y, 0.1)
X_train_val_scal, X_test_scal, y_train_val_scal, y_test_scal = my_l_split(X_scaled, y, 0.1)

# ---------- end split test ----------

# ---------- init split train validation ----------
X_train, X_val, y_train, y_val = my_l_split(X_train_val, y_train_val, 2 / 9)
X_train_scal, X_val_scal, y_train_scal, y_val_scal = my_l_split(X_train_val_scal, y_train_val_scal, 2 / 9)
# ---------- end split train validation ----------



# COUNTING INSTANCES
class_1 = dataset[dataset.ProceduralSuccess == True].count()  # 2983
class_0 = dataset[dataset.ProceduralSuccess == False].count()  # 408
class_1_train = y_train[y_train == True].count()
class_0_train = y_train[y_train == False].count()
class_1_val = y_val[y_val == True].count()
class_0_val = y_val[y_val == False].count()
class_1_test = y_test[y_test == True].count()
class_0_test = y_test[y_test == False].count()

rf = ensemble_random_forest(X_train, y_train)
clf2 = svm_classifier(X_train_scal, y_train_scal)

y_pred_val = rf.predict(X_val)
y_pred_test = rf.predict(X_test)

y_pred_val2 = clf2.predict(X_val_scal)
y_pred_test2 = clf2.predict(X_test_scal)

roc = roc_curve(y_test, y_pred_test)
score = roc_auc_score(y_test, y_pred_test)

roc2 = roc_curve(y_val_scal, y_pred_val2)
score2 = roc_auc_score(y_val_scal, y_pred_val2)

rep = classification_report(y_val, y_pred_val)
rep_test = classification_report(y_test, y_pred_test)

rep2 = classification_report(y_val_scal, y_pred_val2)
rep_test2 = classification_report(y_test_scal, y_pred_test2)
print('Report of RF on test set')
print(rep_test)

print('Report of SVM on test set')
print(rep_test2)

# print('Report of ROC score on val set')
# print(score)

clf = svm.SVC(C=100.0, kernel='rbf', class_weight='balanced', max_iter=-1, random_state=SEED).fit(X_train, y_train)
skf = StratifiedKFold(n_splits=10, random_state=SEED, shuffle=True)
f1_list = []
train_list = []
test_list = []
for train_index, test_index in skf.split(X, y):
    train_list = pd.concat([X.iloc[train_index.tolist(), :], y.iloc[train_index.tolist()]], axis=1)
    test_list = pd.concat([X.iloc[test_index.tolist(), :], y.iloc[test_index.tolist()]], axis=1)
    clf.fit(X.iloc[train_index.tolist(), :], y.iloc[train_index.tolist()])
    y_predicted = clf.predict(X.iloc[test_index.tolist(), :])
    f1_list.append(f1_score(abs(y.iloc[test_index.tolist()] - 1), abs(y_predicted - 1)))
f1_list = np.array(f1_list)
print(f1_list)
print("F1 score (0 class): %0.2f (+/- %0.2f)" % (f1_list.mean(), f1_list.std() * 1.95))

# PLOT PAIRPLOT

# fig = plt.gcf()
# fig.set_size_inches(15, 10)
# sns.set(style="ticks")
# sns.pairplot(dt.loc[:, X.columns.tolist()], hue="ProceduralSuccess")


# PLOT FEATURES CORRELATIONS
# fig = plt.figure()
# axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # left, bottom, width, height (range 0 to 1)#
# axes.plot(roc[0], roc[1])
# axes.set_xlabel('False Positive Rate')
# axes.set_ylabel('True Positive Rate')
# axes.set_title('ROC curve for Random Forest classifier')
# fig.text(0.5, 0.2, r'AUC = %f'%score)
# fig.savefig('ROC_curve_RF.png')
