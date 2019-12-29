# std lib

#  import foo
from sklearn.manifold import TSNE

from ProjectML.general_util import *
from ProjectML.monthDeath.pre_processing import *
from ProjectML.monthDeath.evaluation import *
from ProjectML.monthDeath.classification import *
from ProjectML.monthDeath.feature_processing import *
import ProjectML.monthDeath.plot_confusion_matrix
DT_FILENAME = "dataset/RISPEVA_dataset_for_ML.xlsx"
DT_KNN = "dataset/knn.xlsx"

dt = my_l_extract_feature(DT_KNN,'1 month Death')



# dt = imputation(dt)[0]

dt, dt_test = extract_test(dt,0.22,0.03)
dt_test_X=dt_test.loc[:,'CenterID':'P2Y12 inhibt']
dt_test_y=dt_test.loc[:,'1 month Death']


print(dt.var()<0.5)

X, y, dt = rebalance(dt)

result = feature_importance(X,y,25)
result = (list(list(zip(*result))[0]))
X = X.loc[:, result]

# result=feature_selection(X,y)
# X = X.loc[:, result]



X_train, X_test, y_train, y_test = my_l_split(X,y)
clf = svm_classifier(X_train,y_train)
clf2 = ensemble_random_forest(X_train,y_train)
y_pred_svm=clf.predict(dt_test_X.loc[:,result])
y_pred_random=clf2.predict(dt_test_X.loc[:,result])

predictions = y_pred_random-y_pred_svm
difference = predictions[np.where(predictions!=0)]

#print(report(y_pred_random,y_test))