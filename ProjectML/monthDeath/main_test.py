# Standard lib

# Our import
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import KNNImputer
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.metrics import precision_score, recall_score, f1_score

from ProjectML.general_util import *
from ProjectML.monthDeath.pre_processing.pre_processing import *
from ProjectML.monthDeath.classification import *
from ProjectML.monthDeath.pre_processing.feature_processing import *
from ProjectML.evaluation import *


# Third part lib
from sklearn import svm
from sklearn.preprocessing import Normalizer
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt

DONE_imputation = True
DATASET_FILENAME = '../dataset/RISPEVA_dataset_for_ML.xlsx'
DATASET_IMPUTATION = "../dataset/imputation_1MD.xlsx"
LABEL = '1 month Death'

dataset = my_l_read(DATASET_FILENAME)

death = dataset.loc[:,[LABEL]] == 1
death = death[death==True].dropna()
datset_death = dataset.iloc[death.index.to_list()]

alive = dataset.loc[:,[LABEL]] == 0
alive = alive[alive==True].dropna()
datset_alive = dataset.iloc[alive.index.to_list()]

imputer = KNNImputer(n_neighbors=2, weights="uniform")

dt_knn = dataset.copy(deep=True)
dt_knn.iloc[:, :] = imputer.fit_transform(dataset)
binary_cols = [col for col in dataset if np.isin(dataset[col].dropna().unique(), [0.0, 1.0]).all()]
dt_knn[binary_cols] = dt_knn[binary_cols].round()

X, y ,dataset = my_l_extract_feature(dt_knn, label=LABEL)

features = np.array(X.columns.to_list())

clf = LogisticRegressionCV(max_iter=3000,class_weight='balanced')
y = y.astype('int')
clf.fit(StandardScaler().fit_transform(X), y)
coefs1 = np.abs(clf.coef_[0])
indices1 = np.argsort(coefs1)[::-1]

plt.figure()
plt.title("Feature importances (Logistic Regression)")
plt.bar(range(10), coefs1[indices1[:10]],
       color="r", align="center")
plt.xticks(range(10),features[indices1[:10]], rotation=45, ha='right')
plt.subplots_adjust(bottom=0.3)
plt.show()

clf = RandomForestClassifier(n_jobs=-1, random_state=42, n_estimators=400, max_depth=6, max_features=6,class_weight='balanced') #has already been tuned
clf.fit(X, y)
coefs = clf.feature_importances_
indices = np.argsort(coefs)[::-1]
plt.figure()
plt.title("Feature importances (Random Forests)")
plt.bar(range(10), coefs[indices[:10]],
       color="r", align="center")
plt.xticks(range(10), features[indices[:10]], rotation=45, ha='right')
plt.subplots_adjust(bottom=0.3)

plt.ion(); plt.show()
