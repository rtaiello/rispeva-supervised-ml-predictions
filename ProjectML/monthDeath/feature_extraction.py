# third part
import numpy as np


def drop_corr_feature(X, threshold):
    col_corr = set()  # Set of all the names of deleted columns
    corr_matrix = X.corr()
    dropped_columns = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
                if colname in X.columns:
                    X = X.drop(columns=[colname], axis=1)
                    dropped_columns += [colname]
    return X, dropped_columns


def best_eight_features(X):
    ''' features selected according to XGBoostClassifier features importance'''
    X = X.loc[:, ['CenterID', 'PatientID', 'Age', 'STSScore', 'PeakAorticGradient', 'AorticValveArea', 'AorticAnulus',
                  'Creatinine']]
    return X

BEST_FEATURES = ['Creatinine',
 'modEGFR',
 'EuroSCOREII',
 'Hemoglobin',
 'BMI',
 'SPAP',
 'PeakAorticGradient',
 'SinotubularJunctionDiameter',
 'LogisticEuroSCORE',
 'LVEF',
 'MinAorticAnulus',
 'BSA',
 'STSScore',
 'Albuminemia',
 'MaxAorticAnulus',
 'AorticArchMinDiameter',
 'AscendingAorticMinDiameter',
 'Age',
 'Weight',
 'ValsalvaSinusDiameter',
 'LeftIliacArteryMinDiameter',
 'RightIliacArteryMinDiameter',
 'AorticAnulus',
 'PeakAorticVelocity',
 'ValsalvaSinusDimension',
 'AscendingAorticDimension',
 'RightFemoralArteryMinDiameter',
 'AorticValveArea',
 'MeanAorticGradient',
 'IndexedAorticValveArea',
 'MeanAorticAnulus',
 'Hematocrit',
 'AorticAnulusPerimeter',
 'LeftFemoralArteryMinDiameter']