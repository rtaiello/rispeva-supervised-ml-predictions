import gc

# my lib
from ProjectML.monthDeath.pre_processing import imputation, read_dataset, extract_feature
from ProjectML.general_util.rebalance import *
from ProjectML.general_util.pre_processing import *
from ProjectML.general_util.evaluation import *
from ProjectML.general_util.utility import my_l_concat_series, my_l_concat_dataframe

from ProjectML.monthDeath.feature_extraction import drop_corr_feature

# third part
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

# Constant
IMPUTED = True
DATASET_FILENAME = '../../dataset/RISPEVA_dataset_for_ML.xlsx'
DATASET_FILENAME_CSV = '../../dataset/RISPEVA_1MD_cleaned.csv'
DATASET_IMPUTATION = "../../pickle/1MD/1MD_imputation.pkl"
PATH_PROCESSED = '../../dataset/processed'
LABEL = '1monthDeath'

PERCENTAGE_DROP_FEATURE_CORRELATION = 0.90
TO_DROP = ['PatientID']
if not IMPUTED:
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
    '''
    X_train_imp, std, features_selected = my_l_std_scaling(X_train_imp)
    features_selected = features_selected.columns.to_list()
    X_test_imp.loc[:, features_selected] = std.transform(X_test_imp[features_selected])
    X_val_imp.loc[:, features_selected] = std.transform(X_val_imp[features_selected])
    '''
    # ---------- end scaled features ----------

    dt_train_imp = my_l_concat_series(X_train_imp, y_train)
    dt_test_imp = my_l_concat_series(X_test_imp, y_test)
    dt_val_imp = my_l_concat_series(X_val_imp, y_val)

    dt_train_imp.to_csv(PATH_PROCESSED + "/dt_train_imp.csv", index=False)
    dt_test_imp.to_csv(PATH_PROCESSED + "/dt_test_imp.csv", index=False)
    dt_val_imp.to_csv(PATH_PROCESSED + "/dt_val_imp.csv", index=False)
else:
    dt_train_imp = pd.read_csv(PATH_PROCESSED + "/dt_train_imp.csv")
    dt_test_imp = pd.read_csv(PATH_PROCESSED + "/dt_test_imp.csv")
    dt_val_imp = pd.read_csv(PATH_PROCESSED + "/dt_val_imp.csv")

INTEGER_FEATURES = ["ProsthesisSize1", "AorticValveGradient", "SheathSize", "SPAP"]
dt_train_imp[INTEGER_FEATURES] = dt_train_imp[INTEGER_FEATURES].astype('int32')
dt_test_imp[INTEGER_FEATURES] = dt_test_imp[INTEGER_FEATURES].astype('int32')
dt_val_imp[INTEGER_FEATURES] = dt_val_imp[INTEGER_FEATURES].astype('int32')

# ---------- init log transformation ----------
# "LogisticEuroSCORE", "EuroSCOREII", "EuroSCOREII",'modEGFR'
LOG_FEATURES = ['modEGFR', "LogisticEuroSCORE"]

for feature in LOG_FEATURES:
    # ---------- init log transformation ----------
    log_scaled_feature, log_transformer = my_l_log_scaling(dt_train_imp[feature])
    dt_train_imp["LOG_" + feature] = log_scaled_feature
    dt_test_imp["LOG_" + feature] = log_transformer.transform(dt_test_imp[feature].values.reshape(-1, 1))
    dt_val_imp["LOG_" + feature] = log_transformer.transform(dt_val_imp[feature].values.reshape(-1, 1))
    # ---------- end log transformation ----------
    upper, lower = my_l_percentile_cut_off(dt_train_imp, "LOG_" + feature)
    dt_train_imp = my_l_drop_cut_off(dt_train_imp, "LOG_" + feature, upper=upper, lower=lower)
    # dt_test_imp = my_l_drop_cut_off(dt_test_imp, "LOG_" + feature, upper=upper, lower=lower)
    # dt_val_imp = my_l_drop_cut_off(dt_val_imp, "LOG_" + feature, upper=upper, lower=lower)

    # dt_train_imp.drop(columns="LOG_" + feature inplace=True)
    # dt_test_imp.drop(columns=feature, inplace=True)
    # dt_val_imp.drop(columns=feature, inplace=True)
# ---------- end log transformation ----------

# ---------- init std transformation ----------
# 'BMI',,"BSA",'Hematocrit'
STD_FEATURES = ["Age"]

for feature in STD_FEATURES:
    print("Before the cut of for -> {}".format(feature))
    print(dt_train_imp[feature].describe().T)
    # my_l_plot_feature(dt_train_imp[feature],feature)
    upper, lower = my_l_percentile_cut_off(dt_train_imp, feature,upper=90,lower=0)
    dt_train_imp = my_l_drop_cut_off(dt_train_imp, feature, upper=upper, lower=lower)
    # dt_test_imp = my_l_cut_off(dt_test_imp, feature, upper=upper, lower=lower)
    # dt_val_imp = my_l_cut_off(dt_val_imp, feature, upper=upper, lower=lower)
    print("After the cut of for -> {}".format(feature))
    print(dt_train_imp[feature].describe().T)
# ---------- end log transformation ----------
# FEATURE = "SPAP"
# df_train_imp, age_bucketizer = my_l_bucketizer(dt_train_imp, FEATURE, bins=30)
# dt_test_imp[FEATURE] = age_bucketizer.transform(dt_test_imp.loc[:, [FEATURE]])
# dt_val_imp[FEATURE] = age_bucketizer.transform(dt_val_imp.loc[:, [FEATURE]])

dt_full_imp = my_l_concat_dataframe(df1=dt_train_imp, df2=dt_val_imp)
X_full = dt_full_imp.drop(columns=LABEL)
y_full = dt_full_imp[LABEL]

X_test_imp = dt_test_imp.drop(columns=LABEL)
y_test = dt_test_imp[LABEL]
FEATURES = []  # X_full.columns.to_list()
results = [[value, 0] for value in FEATURES]
for idx, feature in enumerate(FEATURES):
    X_train_imp = dt_train_imp.drop(columns=LABEL)
    X_train_imp = X_train_imp.loc[:, [feature]]
    y_train = dt_train_imp[LABEL]

    X_val_imp = dt_val_imp.drop(columns=LABEL)
    X_val_imp = X_val_imp.loc[:, [feature]]
    y_val = dt_val_imp[LABEL]

    # X_train_imp, y_train = my_l_SMOTTETomek(X_train_imp, y_train, percent=1)

    # dropped_columns = drop_corr_feature(dt_full_imp, PERCENTAGE_DROP_FEATURE_CORRELATION)
    # print(dropped_columns)

    folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

    xgb = XGBClassifier(learning_rate=0.02, n_estimators=20000, max_depth=4, min_child_weight=5, subsample=0.8,
                        colsample_bytree=0.8, objective='binary:logistic',
                        nthread=4, scale_pos_weight=3.5, gamma=0.1, seed=SEED, reg_lambda=1.5, n_jobs=-1)

    xgb.fit(X_train_imp, y_train, eval_set=[(X_val_imp, y_val)], eval_metric='auc', verbose=False,
            early_stopping_rounds=1000)

    pred = xgb.predict_proba(X_test_imp.loc[:, [feature]])
    print(feature)
    results[idx][1] = roc_auc_score(y_test, pred[:, 1])
    print("Final ROC-AUC, computed on the Test dataset: %0.2f " % roc_auc_score(y_test, pred[:, 1]))
FEATURES = X_full.columns.to_list()  # ["Hemoglobin","BMI","LOG_modEGFR"]
X_train_imp = dt_train_imp.drop(columns=LABEL)
X_train_imp = X_train_imp.loc[:, FEATURES]
y_train = dt_train_imp[LABEL]

X_val_imp = dt_val_imp.drop(columns=LABEL)
X_val_imp = X_val_imp.loc[:, FEATURES]
y_val = dt_val_imp[LABEL]

# X_train_imp, y_train = my_l_SMOTTETomek(X_train_imp, y_train,percent=1)


dropped_columns = drop_corr_feature(X_train_imp, PERCENTAGE_DROP_FEATURE_CORRELATION)
print(dropped_columns)

folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

xgb = XGBClassifier(learning_rate=0.02, n_estimators=20000, max_depth=4, min_child_weight=5, subsample=0.8,
                    colsample_bytree=0.8, objective='binary:logistic',
                    nthread=4, scale_pos_weight=3.5, gamma=0.1, seed=SEED, reg_lambda=1.5, n_jobs=-1)

xgb.fit(X_train_imp, y_train, eval_set=[(X_val_imp, y_val)], eval_metric='auc', verbose=False,
        early_stopping_rounds=1000)

pred = xgb.predict_proba(X_test_imp.loc[:, FEATURES])
print("Final ROC-AUC, computed on the Test dataset: %0.2f " % roc_auc_score(y_test, pred[:, 1]))

from xgboost import plot_importance

plot_importance(xgb, color='red', max_num_features=30)

'''
pred = np.zeros(X_test_imp.shape[0])
x = X_full.values
y = y_full.values
x, y = my_l_SMOTTETomek(x, y, percent=1)
val = np.zeros(y.shape[0])

for fold_index, (train_index, val_index) in enumerate(folds.split(x, y)):
  print('Batch {} started...'.format(fold_index))
  gc.collect()
  bst = xgb.fit(x[train_index], y[train_index],
                eval_set=[(x[val_index], y[val_index])],
                early_stopping_rounds=1000,
                verbose=True,
                eval_metric='auc'
                )
  val[val_index] = xgb.predict_proba(x[val_index])[:, 1]
  print('AUC of this val set is {}'.format(roc_auc_score(y[val_index], val[val_index])))
  pred += xgb.predict_proba(X_test_imp.values)[:, 1] / folds.n_splits

print("Final ROC-AUC, computed on the Test dataset: %0.2f " % roc_auc_score(y_test, pred))

xgb.get_booster().get_fscore()
mapper = {'f{0}'.format(i): v for i, v in enumerate(X_full.columns.to_list())}
mapped = {mapper[k]: v for k, v in xgb.get_booster().get_fscore().items()}
from xgboost import plot_importance
plot_importance(mapped, color='red', max_num_features=30)

X_full, y_full = pd.concat([X_train, X_val], axis=1), pd.concat([y_train, y_val], axis=1)
balanced_accuracy_score = get_balanced_accuracy(X_full, y_full, xgb)
print("Balanced accuracy: %0.2f (+/- %0.2f)" % (balanced_accuracy_score.mean(), balanced_accuracy_score.std() * 2))
f1_score = get_f1_scores(X_full, y_full, xgb)
print("f1_score: %0.2f (+/- %0.2f)" % (f1_score.mean(), f1_score.std() * 2))
roc_auc = get_roc_auc(X_full, y_full, xgb)
print("roc_auc: %0.2f (+/- %0.2f)" % (roc_auc.mean(), roc_auc.std() * 2))
'''
