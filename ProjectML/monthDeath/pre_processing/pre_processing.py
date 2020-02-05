# std lib

# my lib
from ProjectML.general_util import *
# third part
from imblearn.over_sampling import RandomOverSampler


# Constant
LABEL = '1 month Death'
SEED = 42

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