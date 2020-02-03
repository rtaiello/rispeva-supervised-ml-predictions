# std lib

#

from ProjectML.general_util import *
from ProjectML.proceduralSuccess.pre_processing import *
from ProjectML.proceduralSuccess.evaluation import *
from ProjectML.proceduralSuccess.classification import *
from ProjectML.proceduralSuccess.feature_processing import *
from sklearn.model_selection import cross_validate
from sklearn.utils.class_weight import compute_class_weight



DT_FILENAME = "../dataset/RISPEVA_dataset_for_ML.xlsx"
DT_KNN = "../dataset/imputed_KNN.xlsx"
LABEL = 'ProceduralSuccess'

dt = my_l_extract_feature(DT_KNN, LABEL)  #3570
#dt = imputation(dt, 'ProceduralSuccess')[0]

X = dt.loc[:, 'CenterID':'P2Y12 inhibt']
y = dt.loc[:, LABEL]



#feature selection with Pearson correlation
cor_support, cor_feature = cor_selector(X, y, 30)
print(str(len(cor_feature)), 'selected features')
print(cor_feature)
X = dt.loc[:, cor_feature]

#feature selection with RFE
# classifier = svm.SVC(kernel='poly', degree=2, random_state=SEED)
# clf = AdaBoostClassifier(base_estimator=classifier, n_estimators=100, algorithm=SAMME)
# rfe_selector = RFE(estimator=clf, n_features_to_select=20, step=10, verbose=5)
# rfe_selector.fit(x_scaled, y)
# rfe_support = rfe_selector.get_support()
# rfe_feature = X.loc[:,rfe_support].columns.tolist()
# print(str(len(rfe_feature)), 'selected features')

x_scaled = StandardScaler().fit_transform(X)

class_1 = dt[dt.ProceduralSuccess == True].count()  #3085
class_0 = dt[dt.ProceduralSuccess == False].count()  #422

X_train_val, X_test, y_train_val, y_test = my_l_split(x_scaled, y, testSize=0.1)
class_1_test = y_test[y_test == True].count()
class_0_test = y_test[y_test == False].count()

X_train, X_val, y_train, y_val = my_l_split(X_train_val, y_train_val, testSize=2/9)
class_1_val = y_val[y_val == True].count()
class_0_val = y_val[y_val == False].count()

#clf = ensemble_random_forest(X_train, y_train)   #best
#clf = ensemble_ada_boosting(X_train,y_train)    #worst
#clf = ensemble_bagging(X_train,y_train)      #medium

clf = svm_classifier(X_train,y_train)

y_pred_stack = clf.predict(X_val)
y_pred = clf.predict(X_test)

rep = classification_report(y_val, y_pred_stack)
rep_test = classification_report(y_test, y_pred)







