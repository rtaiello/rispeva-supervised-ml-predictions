# third part
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.impute import IterativeImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
# constant
from ProjectML.general_util.constant import SEED


def my_l_imp_KNN(dataset):
    knn_imputer = KNNImputer()
    dt_knn = dataset.copy(deep=True)
    dt_knn.iloc[:, :] = knn_imputer.fit_transform(dataset)
    return dt_knn


def my_l_imp_MICE(dataset, strategy="mean"):
    from sklearn.ensemble import RandomForestRegressor

    estimator = KNeighborsRegressor(n_neighbors=15, n_jobs=-1)
    regr = RandomForestRegressor(max_depth=4, random_state=SEED,n_jobs=-1,verbose=1)
    mice_imputer = IterativeImputer(random_state=SEED, max_iter=10, estimator=estimator, verbose=2, initial_strategy =strategy)
    dt_mice = dataset.copy(deep=True)
    print(dt_mice.head())
    dt_mice.iloc[:, :] = mice_imputer.fit_transform(dataset)
    print(dt_mice.head())
    return dt_mice,mice_imputer

def my_l_imp_most_frquent(dataset):
    median_imputer = SimpleImputer(strategy='most_frequent')
    dataset.iloc[:,:] = median_imputer.fit_transform(dataset)
    return dataset,median_imputer

def my_l_imp_mean(dataset):
    mean_imputer = SimpleImputer(strategy='mean')
    dataset.iloc[:,:] = mean_imputer.fit_transform(dataset)
    return dataset,mean_imputer