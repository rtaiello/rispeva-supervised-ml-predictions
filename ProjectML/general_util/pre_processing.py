# os library
import re
import os
import gc

# third part lib
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot  as plt
# constant
from ProjectML.general_util.constant import *


def my_l_read_dataframe(filename: str):
    print(os.getcwd())
    dataset = pd.read_csv(filename) if filename.endswith(".csv") else pd.read_excel(filename)
    my_l_rm_white_space(dataset)
    return dataset


def my_l_rm_white_space(dataset):
    regex = re.compile(r"[\[\]<]", re.IGNORECASE)
    dataset.columns = [regex.sub("_", col) if any(x in str(col) for x in {'[', ']', '<'}) else col for col in
                       dataset.columns.values]
    dataset.columns = dataset.columns.str.replace(' ', '')
    return dataset


def my_l_std_scaling(X):
    feature_selected = X.select_dtypes(include=['int64', 'float64'])
    feature_selected = feature_selected
    X_scaled = X.copy()
    std_transformer = StandardScaler()
    X_scaled[feature_selected.columns] = std_transformer.fit_transform(X_scaled[feature_selected.columns])
    return X_scaled, std_transformer, feature_selected


def my_l_norm_scaling(X):
    feature_selected = X.select_dtypes(include=['int64', 'float64'])
    X_scaled = X.copy()
    X_scaled[feature_selected.columns] = Normalizer().fit_transform(X_scaled[feature_selected.columns])
    return X_scaled

def my_l_log_scaling(feature):
  from sklearn.preprocessing import PowerTransformer

  log_trasformer = PowerTransformer() # log transformation
  target_feature = feature.to_numpy().reshape(-1,1)


  log_trasformer.fit(target_feature)
  log_scaled_feature = log_trasformer.transform(target_feature)
  return log_scaled_feature,log_trasformer

def my_l_split(X, y, split_percent):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_percent, random_state=SEED,
                                                        stratify=y, shuffle=True)
    return X_train, X_test, y_train, y_test


def my_l_plot_feature(feature, label_ax, path=None):
    def ecdf(data):
        import numpy as np
        # Empirical Cumulative Distribution Function
        """Compute ECDF for a one-dimensional array of measurements."""
        # Number of data points: n
        n = len(data)
        # x-data for the ECDF: x
        x = np.sort(data)
        # y-data for the ECDF: y
        y = np.arange(1, n + 1) / n
        return x, y

    x_amt_credit, y_amt_credit = ecdf(feature)

    # Generate plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('ECDF vs histogram:' + label_ax)
    ax1.plot(x_amt_credit, y_amt_credit, marker='.', linestyle='none', color='red')
    ax1.set(xlabel=label_ax, ylabel='ECDF')

    feature.hist(ax=ax2)
    ax2.set(xlabel=label_ax, ylabel='Frequency')

    # Display the plot
    if path:
        plt.savefig(path)
    else:
        plt.show()


def my_l_one_hot_encoding(df, categories):
    from sklearn.preprocessing import OneHotEncoder

    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(df[categories])
    df = my_l_apply_ohe(df, encoder=enc, categories = categories)

    df_train = my_l_rm_white_space(df)
    return df, enc


def my_l_apply_ohe(df, encoder, categories):
    import numpy as np
    transformed_train = encoder.transform(df[categories])
    ohe_columns = []
    enc_categories = [i.tolist() for i in encoder.categories_]
    for i in range(len(categories)):
        s = categories[i]
        output = ["{}_{}".format(i, s) for i in enc_categories[i]]
        ohe_columns += output
    ohe_columns = [("is_{}".format(str(x))).upper() for x in ohe_columns]

    ohe_df = pd.DataFrame(transformed_train.toarray(), columns=ohe_columns)
    del transformed_train
    gc.collect()
    df.reset_index(drop=True, inplace=True)
    ohe_df.reset_index(drop=True, inplace=True)

    df = pd.concat([df, ohe_df], axis=1).drop(categories, axis=1)

    binary_cols = [col for col in df if np.isin(df[col].dropna().unique(), [0, 1]).all()]
    df[binary_cols] = df[binary_cols].astype('bool')

    return df

def my_l_percentile_cut_off(df, feature):
    import numpy as np
    ninthyseven_ci = np.percentile(df[feature], [2.5, 97.5])
    mean = np.mean(df[feature])
    cut_off_lower = ninthyseven_ci[0]
    cut_off_upper = ninthyseven_ci[1]
    lower, upper = cut_off_lower, cut_off_upper
    upper_cutted = len(df[(df[feature] > upper) == True])
    lower_cutted = len(df[(df[feature]< lower) == True])
    print("Number of samples ruled out, cut_off_lower {:d}, cut_of_upper {:d}".format(lower_cutted, upper_cutted))
    return upper, lower

def my_l_cut_off(df,feature,upper,lower):
    df.loc[df[feature] < lower, feature] = lower
    df.loc[df[feature] > upper, feature] = upper
    return df