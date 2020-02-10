# std lib

# my lib
from ProjectML.general_util import *
# third part
from imblearn.over_sampling import RandomOverSampler


# Constant
LABEL = '1 month Death'
SEED = 42

def imputation(dataset):
    #NON RIMUOVO LE ROW CON LA LABEL NAN

    #RIMUOVO LE FEATURE CON IL NULL >= 60%
    d_month_mean_null = dataset.isnull().mean() * 100
    col_removed = d_month_mean_null[d_month_mean_null >= 60.0].index.tolist()
    dataset = dataset.drop(columns=col_removed)
    binary_cols = [col for col in dataset if np.isin(dataset[col].dropna().unique(), [0, 1]).all()]
    dataset[binary_cols] = dataset[binary_cols].astype('bool')
    interger_cols = dataset.select_dtypes(include=['int64'])
    dataset = my_l_imp_KNN(dataset)
    dataset[binary_cols] = dataset[binary_cols].round()
    dataset[binary_cols] = dataset[binary_cols].astype('bool')
    dataset[interger_cols.columns] = dataset[interger_cols.columns].astype('int64')
    return dataset

def rebalance(dataset, percent):
    X = dataset.loc[:, 'CenterID':'P2Y12 inhibt']
    y = dataset.loc[:, LABEL]
    if percent >0:
        dataset, X, y = my_l_rebalance(X, y, percent)
    return X, y, dataset

def over_sampling(X,y):
    ros = RandomOverSampler(random_state=SEED, sampling_strategy=0.5)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    from collections import Counter
    print(sorted(Counter(y_resampled).items()))
    return X_resampled,y_resampled
