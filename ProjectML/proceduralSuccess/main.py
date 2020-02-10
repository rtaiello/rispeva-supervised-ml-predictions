# std lib

#

from ProjectML.general_util import *
from ProjectML.proceduralSuccess.pre_processing import *
from ProjectML.proceduralSuccess.evaluation import *
from ProjectML.proceduralSuccess.classification import *
from ProjectML.proceduralSuccess.feature_processing import *
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from numpy import mean
import seaborn as sns
import matplotlib.pyplot as plt
import re





DT_FILENAME = "../dataset/RISPEVA_dataset_for_ML.xlsx"
DT_KNN = "../dataset/imputed_KNN.xlsx"
LABEL = 'ProceduralSuccess'

dt = my_l_extract_feature(DT_KNN, LABEL)  #3391
#dt = imputation(dt, 'ProceduralSuccess')[0]

X = dt.loc[:, 'CenterID':'P2Y12 inhibt']
y = dt.loc[:, LABEL]

#feature selection with Pearson correlation
# cor_support, cor_feature, cor_list = cor_selector(X, y, 40)
# print(str(len(cor_feature)), 'selected features')
# print(cor_feature)
# X = dt.loc[:, cor_feature]
# print(cor_list)

#feature selection with RFE
# clf = RandomForestClassifier(n_estimators=300, criterion='gini',max_features=15, n_jobs=-1, class_weight='balanced', random_state=SEED)
# rfe_selector = RFE(estimator=clf, n_features_to_select=40, step=10, verbose=5)
# rfe_selector.fit(X, y)
# rfe_support = rfe_selector.get_support()
# rfe_feature = X.loc[:,rfe_support].columns.tolist()
# print(str(len(rfe_feature)), 'selected features')

#mask = voting_feature_selection(X,y)
#X = X.loc[:, mask]

class_1 = dt[dt.ProceduralSuccess == True].count()  #2983
class_0 = dt[dt.ProceduralSuccess == False].count()  #408

#TEST SET
X_train_val, X_test, y_train_val, y_test = my_l_split(X, y, testSize=0.1)
class_1_test = y_test[y_test == True].count()
class_0_test = y_test[y_test == False].count()

X_train, X_val, y_train, y_val = my_l_split(X_train_val, y_train_val, testSize=2/9)

#new_df, X_train, y_train = my_l_rebalance(X_train, y_train, percent=0.50)

class_1_train = y_train[y_train == True].count()
class_0_train = y_train[y_train == False].count()

class_1_val = y_val[y_val == True].count()
class_0_val = y_val[y_val == False].count()

#regex = re.compile(r"\[|\]|<", re.IGNORECASE)
#X_train.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in X_train.columns.values]

# model = XGBClassifier()
# # define grid
# weights = [1, 7, 10, 25, 50, 75, 99, 100, 1000]
# param_grid = dict(scale_pos_weight=weights)
# # define evaluation procedure
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# # define grid search
# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='f1')
# # execute the grid search
# grid_result = grid.fit(X_train, abs(y_train-1))
# # report the best configuration
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# # report all configurations
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))


#new_df = X_train.select_dtypes(include=['float64'])

#new_df = StandardScaler().fit_transform(new_df)

#cols_to_scale = new_df.columns

#X_train_scaled = scaleColumns(X_train.copy(),cols_to_scale)

#clf = ensemble_random_forest(X_train, y_train)   #best
#clf = ensemble_ada_boosting(X_train,y_train)    #worst
#clf = svm_classifier(X_train,y_train)
#clf = ensemble_bagging(X_train,y_train)      #medium

# y_pred_val = clf.predict(X_val)
# y_pred_test = clf.predict(X_test)
#
# roc = roc_curve(y_val, y_pred_val)
# score = roc_auc_score(y_val, y_pred_val)
#
# rep = classification_report(y_val, y_pred_val)
# rep_test = classification_report(y_test, y_pred_test)
# print('Report of classifier on val set')
# print(rep)
# print('Report of ROC score on val set')
# print(score)
clf = RandomForestClassifier(n_estimators=500, criterion='gini', max_features=5, n_jobs=-1, class_weight='balanced',
                               random_state=SEED)
skf = StratifiedKFold(n_splits=10)
f1_list=[]
train_list = []
test_list = []
for train_index, test_index in skf.split(X_train_val, y_train_val):
    train_list = pd.concat([X_train_val.iloc[train_index.tolist(), :], y_train_val.iloc[train_index.tolist()]],axis=1)
    test_list = pd.concat([X_train_val.iloc[test_index.tolist(), :],y_train_val.iloc[test_index.tolist()]],axis=1)
    clf.fit(X_train_val.iloc[train_index.tolist(), :], y_train_val.iloc[train_index.tolist()])
    y_predicted = clf.predict(X_train_val.iloc[test_index.tolist(), :])
    f1_list.append(f1_score(abs(y_train_val.iloc[test_index.tolist()]-1), abs(y_predicted-1)))
print(f1_list)




#PLOT PAIRPLOT
# fig = plt.gcf()
# fig.set_size_inches(15, 10)
# sns.set(style="ticks")
# sns.pairplot(dt.loc[:, [' Steno-insufficienza Valvolare Aortica','Cancer','ProceduralSuccess']], hue="ProceduralSuccess")


#PLOT FEATURES CORRELATIONS
# fig = plt.figure()
# axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # left, bottom, width, height (range 0 to 1)#
# axes.plot(roc[0], roc[1])
# axes.set_xlabel('fpr')
# axes.set_ylabel('tpr')
# axes.set_title('ROC curve for SVM classifier')
# fig.text(0.5, 0.2, r'AUC = %f'%score)
# fig.savefig('ROC_curve_svm')

# Update plot object with X/Y axis labels and Figure Title
#plt.xlabel(X.columns[0], size=14)
#plt.ylabel(X.columns[1], size=14)
#plt.title('SVM Decision Region Boundary', size=16)
#plt.show()




