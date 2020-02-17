# third part
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer
from sklearn.impute import IterativeImputer

# constant
from ProjectML.general_util.constant import SEED


def my_l_imp_KNN(dataset):
    knn_imputer = KNNImputer()
    dt_knn = dataset.copy(deep=True)
    dt_knn.iloc[:, :] = knn_imputer.fit_transform(dataset)
    return dt_knn


def my_l_imp_MICE(dataset):
    mice_imputer = IterativeImputer(random_state=SEED)
    dt_mice = dataset.copy(deep=True)
    dt_mice.iloc[:, :] = mice_imputer.fit_transform(dataset)
    return dt_mice
