import gc

# my lib
from ProjectML.monthDeath.pre_processing import imputation, read_dataset, extract_feature

from ProjectML.monthDeath.classification import *
from ProjectML.general_util.pre_processing import *
from ProjectML.general_util.evaluation import *
from ProjectML.general_util.utility import my_l_concat

# third part
from sklearn.metrics import roc_auc_score

# Constant
DONE_imputation = True
DATASET_FILENAME = '../../dataset/RISPEVA_dataset_for_ML.xlsx'
DATASET_FILENAME_CSV = '../../dataset/RISPEVA_1MD_cleaned.csv'
DATASET_IMPUTATION = "../../pickle/1MD/1MD_imputation.pkl"
LABEL = '1monthDeath'
dataset = None

PERCENTAGE_DROP_FEATURE_CORRELATION = 0.6
TO_DROP = ['PatientID']

dataset = read_dataset(DATASET_FILENAME)
dataset = dataset[dataset['1monthDeath'].notnull()]
dataset.drop(columns=TO_DROP, inplace=True)

X, y, dataset = extract_feature(dataset)

# ----------  end features selection----------

# ---------- init split test ----------
X, X_test, y, y_test = my_l_split(X, y, split_percent=0.2)
# ---------- end split test ----------

# ---------- init split train validation ----------
X_train, X_val, y_train, y_val = my_l_split(X, y, split_percent=1 / 9)
# ---------- end split train validation ----------

# ---------- init imputation ----------
X_train_imp, col_removed, imputer = imputation(X_train)

X_test_imp = X_test.copy(deep=True)
X_val_imp = X_val.copy(deep=True)

X_test_imp.drop(columns=col_removed, inplace=True)
X_test_imp.iloc[:, :] = imputer.transform(X_test_imp)
X_val_imp.drop(columns=col_removed, inplace=True)
X_val_imp.iloc[:, :] = imputer.transform(X_val_imp)

# ---------- end imputation ----------

# ---------- init one hot encoding ----------
CATEGORICAL_FEATURES = ["CenterID", "ChronicCardiacSyndrome"]
X_train_imp, enc_transformer = my_l_one_hot_encoding(df=X_train_imp, categories=CATEGORICAL_FEATURES)
X_test_imp = my_l_apply_ohe(X_test_imp, enc_transformer, CATEGORICAL_FEATURES)
X_val_imp = my_l_apply_ohe(X_val_imp, enc_transformer, CATEGORICAL_FEATURES)
# ---------- end one hot encoding ----------

# ---------- init scaled features ----------
X_train_imp, std, features_selected = my_l_std_scaling(X_train_imp)
features_selected = features_selected.columns.to_list()
X_test_imp.loc[:, features_selected] = std.transform(X_test_imp[features_selected])
X_val_imp.loc[:, features_selected] = std.transform(X_val_imp[features_selected])
# ---------- end scaled features ----------

dt_train_imp = my_l_concat(X_train_imp, y_train)
dt_test_imp = my_l_concat(X_test_imp, y_test)
dt_val_imp = my_l_concat(X_val_imp, y_val)
TARGET = "LogisticEuroSCORE"

# ---------- init log transformation ----------
log_scaled_feature, log_transformer = my_l_log_scaling(dt_train_imp[TARGET])
dt_train_imp["LOG_" + TARGET] = log_scaled_feature
dt_test_imp["LOG_" + TARGET] = log_transformer.transform(dt_test_imp[TARGET].values.reshape(-1, 1))
dt_val_imp["LOG_" + TARGET] = log_transformer.transform(dt_val_imp[TARGET].values.reshape(-1, 1))
# ---------- end log transformation ----------

upper, lower = my_l_percentile_cut_off(dt_train_imp, "LOG_" + TARGET)
dt_train_imp = my_l_cut_off(dt_train_imp, "LOG_" + TARGET, upper=upper, lower=lower)
dt_test_imp = my_l_cut_off(dt_test_imp, "LOG_" + TARGET, upper=upper, lower=lower)
dt_val_imp = my_l_cut_off(dt_val_imp, "LOG_" + TARGET, upper=upper, lower=lower)

dt_train_imp.drop(columns=TARGET, inplace=True)
dt_test_imp.drop(columns=TARGET, inplace=True)
dt_val_imp.drop(columns=TARGET, inplace=True)
3 / 0
print("Percent of death in original dataset= {0:.2f}".format(y[y == 1].count() / y.count()))
print("Percent of death in test set= {0:.2f}".format(y_test[y_test == 1].count() / y_test.count()))
print("Percent of death in validation set= {0:.2f}".format(y_val[y_val == 1].count() / y_val.count()))

xgb = XGBClassifier(learning_rate=0.02, n_estimators=20000, max_depth=4, min_child_weight=5, subsample=0.8,
                    colsample_bytree=0.8, objective='binary:logistic',
                    nthread=4, scale_pos_weight=4, gamma=0.1, seed=SEED, reg_lambda=1.5, n_jobs=-1)

xgb.fit(X_train_imp, y_train, eval_set=[(X_val_imp, y_val)], eval_metric='auc', verbose=True,
        early_stopping_rounds=1000)
y_pred_proba = xgb.predict_proba(X_test_imp)
print("FINAL roc_auc: %0.2f " % roc_auc_score(y_test, y_pred_proba[:, 1]))

xgb.get_booster().get_fscore()
mapper = {'f{0}'.format(i): v for i, v in enumerate(X_train_imp.columns.to_list())}
mapped = {mapper[k]: v for k, v in xgb.get_booster().get_fscore().items()}
from xgboost import plot_importance

plot_importance(mapped, color='red', max_num_features=30)

X_train_imp.reset_index(drop=True, inplace=True)
X_val_imp.reset_index(drop=True, inplace=True)

y_train.reset_index(drop=True, inplace=True)
y_val.reset_index(drop=True, inplace=True)

X_full, y_full = pd.concat([X_train_imp, X_val_imp], axis=0), pd.concat([y_train, y_val], axis=0)

val = np.zeros(y_full.shape[0])
pred = np.zeros(X_test.shape[0])
x = X_full.values
y = y_full.values
from sklearn.model_selection import StratifiedKFold

folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

for fold_index, (train_index, val_index) in enumerate(folds.split(x, y)):
    print('Batch {} started...'.format(fold_index))
    gc.collect()
    bst = xgb.fit(x[train_index], y[train_index],
                  eval_set=[(x[val_index], y[val_index])],
                  early_stopping_rounds=1000,
                  verbose=200,
                  eval_metric='auc'
                  )
    val[val_index] = xgb.predict_proba(x[val_index])[:, 1]
    print('auc of this val set is {}'.format(roc_auc_score(y[val_index], val[val_index])))
    pred += xgb.predict_proba(X_test_imp.values)[:, 1] / folds.n_splits

'''
X_full, y_full = pd.concat([X_train, X_val], axis=1), pd.concat([y_train, y_val], axis=1)
balanced_accuracy_score = get_balanced_accuracy(X_full, y_full, xgb)
print("Balanced accuracy: %0.2f (+/- %0.2f)" % (balanced_accuracy_score.mean(), balanced_accuracy_score.std() * 2))
f1_score = get_f1_scores(X_full, y_full, xgb)
print("f1_score: %0.2f (+/- %0.2f)" % (f1_score.mean(), f1_score.std() * 2))
roc_auc = get_roc_auc(X_full, y_full, xgb)
print("roc_auc: %0.2f (+/- %0.2f)" % (roc_auc.mean(), roc_auc.std() * 2))

xgb = xgb_classifier(X_full, y_full)

'''
