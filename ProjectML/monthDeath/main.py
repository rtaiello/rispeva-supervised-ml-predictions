# my lib
from ProjectML.monthDeath.pre_processing import imputation, read_dataset, extract_feature

from ProjectML.monthDeath.feature_extraction import *
from ProjectML.monthDeath.classification import *
from ProjectML.general_util.pre_processing import *
from ProjectML.general_util.evaluation import *

# third part
import pandas as pd

# Constant
DONE_imputation = True
DATASET_FILENAME = '../../dataset/RISPEVA_dataset_for_ML.xlsx'
DATASET_FILENAME_CSV = '../../dataset/RISPEVA_1MD_cleaned.csv'
DATASET_IMPUTATION = "../../pickle/1MD/1MD_imputation.pkl"
LABEL = '1monthDeath'
dataset = None

PERCENTAGE_DROP_FEATURE_CORRELATION = 0.6
def one_hot_encoding(df_train, categories):
    import gc
    import uuid
    from sklearn.preprocessing import OneHotEncoder

    enc = OneHotEncoder(handle_unknown='ignore')

    enc.fit(df_train[categories])
    transformed_train = enc.transform(df_train[categories])
    ohe_columns = [item for sub_array in enc.categories_ for item in sub_array.tolist()]
    ohe_columns = [("is_{}".format(str(x))).upper() for x in ohe_columns]

    ohe_df = pd.DataFrame(transformed_train.toarray(), columns=ohe_columns)
    del transformed_train
    gc.collect()
    df_train.reset_index(drop=True, inplace=True)
    ohe_df.reset_index(drop=True, inplace=True)

    df_train = pd.concat([df_train, ohe_df], axis=1).drop(categories, axis=1)
    # df_train[ohe_columns] = df_train[ohe_columns].astype('bool')

    df_train = my_l_rm_white_space(df_train)
    return df_train, enc, ohe_columns



# ---------- init imputation ----------
dataset = read_dataset(DATASET_FILENAME)
dataset = dataset[dataset['1monthDeath'].notnull()]

X, y, dataset = extract_feature(dataset)


# ----------  end features selection----------

# ---------- init split test ----------
X, X_test, y, y_test = my_l_split(X, y, split_percent=0.1)
# ---------- end split test ----------

# ---------- init split train validation ----------
X_train, X_val, y_train, y_val = my_l_split(X, y, split_percent=2 / 9)

X_train, col_removed, imputer = imputation(X_train)

X_test.drop(columns=col_removed, inplace=True)

X_val.drop(columns=col_removed, inplace=True)

X_test.iloc[:,:] = imputer.transform(X_test)

X_val.iloc[:,:] = imputer.transform(X_val)
CATEGORICAL_FEATURES = ["CenterID"]
X_train, enc_trasformer, ohe_columns = one_hot_encoding(df_train=X_train, categories=CATEGORICAL_FEATURES)
X_val_mice = X_val
transformed_test = enc_trasformer.transform(X_val_mice[CATEGORICAL_FEATURES])
ohe_df = pd.DataFrame(transformed_test.toarray(), columns=ohe_columns)
X_val_mice.reset_index(drop=True, inplace=True)
ohe_df.reset_index(drop=True, inplace=True)
X_val_mice_enc =  pd.concat([X_val_mice, ohe_df], axis=1).drop(CATEGORICAL_FEATURES, axis=1)
#df_train[ohe_columns] = df_test[ohe_columns].astype('bool')
X_val_mice_enc = my_l_rm_white_space(X_val_mice_enc)
X_val = X_val_mice_enc

X_val_mice = X_test
transformed_test = enc_trasformer.transform(X_val_mice[CATEGORICAL_FEATURES])
ohe_df = pd.DataFrame(transformed_test.toarray(), columns=ohe_columns)
X_val_mice.reset_index(drop=True, inplace=True)
ohe_df.reset_index(drop=True, inplace=True)
X_val_mice_enc =  pd.concat([X_val_mice, ohe_df], axis=1).drop(CATEGORICAL_FEATURES, axis=1)
#df_train[ohe_columns] = df_test[ohe_columns].astype('bool')
X_val_mice_enc = my_l_rm_white_space(X_val_mice_enc)
X_test = X_val_mice_enc
# ---------- end split train validation ----------

print("Percent of death in original dataset= {0:.2f}".format(y[y == 1].count() / y.count()))
print("Percent of death in test set= {0:.2f}".format(y_test[y_test == 1].count() / y_test.count()))
print("Percent of death in validation set= {0:.2f}".format(y_val[y_val == 1].count() / y_val.count()))

print("-DROPPED FEATURES WITH CORR GRATER THAN 60 - FEATURES IMPORTANCE: THE BEST 8 - IMBALANCED DATASET-")
xgb = xgb_classifier(X_train, y_train)
y_pred = xgb.predict(X_val)
print(report(y_val, y_pred))
print("Cross validation cv = 5 ")
X_full, y_full = pd.concat([X_train, X_val], axis=0), pd.concat([y_train, y_val], axis=0)
balanced_accuracy_score = get_balanced_accuracy(X_full, y_full, xgb)
print("Balanced accuracy: %0.2f (+/- %0.2f)" % (balanced_accuracy_score.mean(), balanced_accuracy_score.std() * 2))
f1_score = get_f1_scores(X_full, y_full, xgb)
print("f1_score: %0.2f (+/- %0.2f)" % (f1_score.mean(), f1_score.std() * 2))
roc_auc = get_roc_auc(X_full, y_full, xgb)
print("roc_auc: %0.2f (+/- %0.2f)" % (roc_auc.mean(), roc_auc.std() * 2))

xgb = xgb_classifier(X_full, y_full)
y_pred_proba = xgb.predict_proba(X_test)
from sklearn.metrics import roc_auc_score
print("FINAL roc_auc: %0.2f " % roc_auc_score(y_test, y_pred_proba[:,1]))

