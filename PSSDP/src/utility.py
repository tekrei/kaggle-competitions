import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from seaborn import heatmap
from sklearn.metrics import make_scorer


def change_datatype(df):
    for column in df.columns:
        mx = df[column].max()
        mn = df[column].min()
        # reduce memory size of float columns
        if(df[column].dtype == np.float):
            if mn > np.finfo(np.float16).min and mx < np.finfo(np.float16).max:
                df[column] = df[column].astype(np.float16)
            elif mn > np.finfo(np.float32).min and mx < np.finfo(np.float32).max:
                df[column] = df[column].astype(np.float32)
            else:
                df[column] = df[column].astype(np.float64)
        # reduce memory size of int columns
        if(df[column].dtype == np.int):
            if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                df[column] = df[column].astype(np.int8)
            elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                df[column] = df[column].astype(np.int16)
            elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                df[column] = df[column].astype(np.int32)
            else:
                df[column] = df[column].astype(np.int64)
    # print("Size: %d" % df.memory_usage(deep=True).sum())

# Define the gini metric - from https://www.kaggle.com/c/ClaimPredictionChallenge/discussion/703#5897




#Remove redundant calls
def ginic(actual, pred):
    actual = np.asarray(actual) #In case, someone passes Series or list
    n = len(actual)
    a_s = actual[np.argsort(pred)]
    a_c = a_s.cumsum()
    giniSum = a_c.sum() / a_s.sum() - (n + 1) / 2.0
    return giniSum / n

def gini_normalizedc(a, p):
    if p.ndim == 2:#Required for sklearn wrapper
        p = p[:,1] #If proba array contains proba for both 0 and 1 classes, just pick class 1
    return ginic(a, p) / ginic(a, a)


def gini(actual, pred, cmpcol=0, sortcol=1):
    assert(len(actual) == len(pred))
    all = np.asarray(
        np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)


def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)


def gini_xgb(preds, dtrain):
    # Create an XGBoost-compatible metric from Gini
    labels = dtrain.get_label()
    # gini_score = gini_normalized(labels, preds)
    gini_score = gini_normalizedc(labels, preds)
    return [('gini', gini_score)]

def gini_lgb(preds, dtrain):
    # Create an XGBoost-compatible metric from Gini
    labels = dtrain.get_label()
    # gini_score = gini_normalized(labels, preds)
    gini_score = gini_normalizedc(labels, preds)
    return [('gini', gini_score, True)]


gini_scorer = make_scorer(gini_normalized, greater_is_better=True)


def plot_heatmap(corr):
    heatmap(corr, xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)


def plot_corr(corr, size=10):
    fig, ax = plt.subplots(figsize=(size, size))
    ax = fig.add_subplot(111)
    cax = ax.matshow(corr, vmin=-1, vmax=1, cmap='bwr')
    fig.colorbar(cax)
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)


def preprocess(df_train, df_test):
    print("Before preprocessing %s %s "%(df_train.values.shape, df_test.values.shape))

    # drop ps_calc_ columns
    col_to_drop = df_train.columns[df_train.columns.str.startswith('ps_calc_')]
    df_train = df_train.drop(col_to_drop, axis=1)
    df_test = df_test.drop(col_to_drop, axis=1)

    # one hot encodable columns: 'ps_ind_01', 'ps_ind_03', 'ps_ind_15', 'ps_reg_02', 'ps_car_04_cat', 'ps_car_06_cat', 'ps_car_15'
    # ohe_columns = ['ps_ind_01', 'ps_ind_03', 'ps_ind_15', 'ps_reg_02', 'ps_car_04_cat', 'ps_car_06_cat', 'ps_car_15']

    cat_features = [a for a in df_train.columns if a.endswith('cat')]

    df_train = pd.get_dummies(df_train, columns=cat_features)
    df_test = pd.get_dummies(df_test, columns=cat_features)

    # columns with very skewed data ['ps_ind_10_bin','ps_ind_11_bin','ps_ind_12_bin','ps_ind_13_bin'].
    # columns with outliers ['ps_ind_14', 'ps_car_10_cat']
    # drop low covariance columns
    # drop_set = ['ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin', 'ps_ind_13_bin', 'ps_ind_14', 'ps_ind_18_bin', 'ps_car_10_cat', 'ps_car_11_cat']
    # df_train = df_train.drop(drop_set, axis=1)
    # df_test = df_test.drop(drop_set, axis=1)

    print("After preprocessing %s %s "%(df_train.values.shape, df_test.values.shape))
    return df_train, df_test

def read_data():
    # Read in our input data - reading missing values as -1
    train_df = pd.read_csv('../input/train.csv', na_values="-1")
    change_datatype(train_df)
    test_df = pd.read_csv('../input/test.csv', na_values="-1")
    change_datatype(test_df)
    # fill NA values with mean
    # train_df = train_df.fillna(train_df.mean())
    # test_df = test_df.fillna(test_df.mean())
    return train_df, test_df
