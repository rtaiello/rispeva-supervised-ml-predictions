# third part lib
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

# constant
SEED = 42


def my_l_resample(X, y, ros):
    X_res, y_res = ros.fit_resample(X, y)
    from collections import Counter
    print(sorted(Counter(y_res).items()))
    return X_res, y_res


def my_l_under_sampling(X, y, percent):
    ros = RandomUnderSampler(random_state=SEED, sampling_strategy=percent)
    return my_l_resample(X, y, ros)

def my_l_SMOTE_sampling(X, y, percent):
    ros = SMOTE(random_state=SEED, sampling_strategy=percent)
    return my_l_resample(X, y, ros)
def my_l_ADASYN_sampling(X,y,percent):
    ros = ADASYN(random_state=SEED,sampling_strategy=percent)
    return my_l_resample(X,y,ros)