# Standard lib

# Our import
from sklearn.ensemble import GradientBoostingRegressor

from ProjectML.general_util import *
from ProjectML.monthDeath.pre_processing.pre_processing import *
from ProjectML.monthDeath.classification import *
from ProjectML.monthDeath.pre_processing.feature_processing import *
from ProjectML.evaluation import *


# Third part lib
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Lasso, LassoCV

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
X, y, dataset = rebalance(dataset, percent=0)
# ---------- end rebalance ----------




X_train, X_test, y_train, y_test = my_l_split(X, y)

# ---------- init f. importance ----------
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
lcv = LassoCV()
lcv.fit(X_train_std,y_train)
lcv_mask = lcv.coef_ !=0
n_features = sum(lcv_mask)

rfe_rf = RFE(estimator=RandomForestRegressor(),
n_features_to_select=n_features, step=5, verbose=1)
rfe_rf.fit(X_train, y_train)
rf_mask = rfe_rf.support_

rfe_gb = RFE(estimator=GradientBoostingRegressor(),
n_features_to_select=n_features, step=5, verbose=1)
rfe_gb.fit(X_train, y_train)
gb_mask = rfe_gb.support_

votes = np.sum([lcv_mask, rf_mask, gb_mask], axis=0)

mask = votes >= 2
X_train, X_test = X_train.loc[:,mask], X_test.loc[:,mask]
# ---------- end f. importance ----------

# ---------- init Classifiers ----------
svm = svm_classifier(X_train, y_train)
random_forest = ensemble_random_forest(X_train, y_train)
ada_boost = ensemble_ada_boosting(X_train, y_train)
mlp = MLPClassifier(max_iter=200,verbose=True, hidden_layer_sizes=(100, ),batch_size=100)
mlp.fit(X_train, y_train)
# ---------- end Classifiers ----------

# ---------- init SVM ----------
y_pred_svm=svm.predict(X_test)
# ---------- end SVM ----------

# ---------- init Random Forest ----------
y_pred_random_forest=random_forest.predict(X_test)
# ---------- end Random Forest ----------

# ---------- init Boosting ----------
y_pred_boosting=ada_boost.predict(X_test)
# ---------- end Boosting ----------

# ---------- init MLP ----------
y_pred_mlp=mlp.predict(X_test)
# ---------- end MLP ----------

print("SVM's prediction:")
print(report(y_test, y_pred_svm))

print("Random Forest's prediction")
print(report(y_test, y_pred_random_forest))

print("Ada Boost's prediction")
print(report(y_test, y_pred_boosting))

print("MLP's prediction")
print(report(y_test, y_pred_mlp))