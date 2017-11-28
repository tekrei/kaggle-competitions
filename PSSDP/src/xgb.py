import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost.sklearn import XGBClassifier
import gc


def multiple_run(model, dtrain, predictors, cv_folds=4, early_stopping_rounds=100):
    print("running a multiple fit with CV to check predictors")
    xgtrain = xgb.DMatrix(dtrain[predictors].values,
                          label=dtrain['target'].values)
    cvresult = xgb.cv(model.get_xgb_params(), xgtrain, num_boost_round=model.get_params()['n_estimators'], nfold=cv_folds,
                      early_stopping_rounds=early_stopping_rounds, verbose_eval=10, feval=gini_xgb, maximize=True)
    gc.collect()
    # fit the algorithm on the data
    model.fit(dtrain[predictors], dtrain['target'])

    # plot feature importances
    fig, ax = plt.subplots(figsize=(20, 20))
    xgb.plot_importance(model, ax=ax)
    plt.savefig("importance.pdf")

    # plot tree
    fig, ax = plt.subplots(figsize=(20, 20))
    xgb.plot_tree(model,ax=ax)
    plt.savefig("tree.pdf")


def single_run(params, x_train, y_train, d_test):
    print("running a single fit and predict with parameters and save results")
    id_test = test_df['id'].values
    # Take a random 20% of the dataset as validation data
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train, y_train, test_size=0.35, random_state=4242)

    # Convert our data into XGBoost format
    d_train = xgb.DMatrix(x_train, y_train)
    d_valid = xgb.DMatrix(x_valid, y_valid)
    d_test = xgb.DMatrix(x_test)
    # This is the data xgboost will test on after eachboosting round
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]

    # maximum of 10,000 rounds with early stopping after 100
    # and the custom metric (maximize=True tells xgb that higher metric is better)
    model = xgb.train(params, d_train, 20000, watchlist, early_stopping_rounds=200,
                      feval=gini_xgb, maximize=True, verbose_eval=50)

    # Predict on our test data
    p_test = model.predict(d_test)

    # Create a submission file
    submission = pd.DataFrame()
    submission['id'] = id_test
    submission['target'] = p_test
    submission.to_csv('submission.csv', index=False)


# Define the gini metric - from https://www.kaggle.com/c/ClaimPredictionChallenge/discussion/703#5897
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
    gini_score = gini_normalized(labels, preds)
    return [('gini', gini_score)]


def grid_search(model, df_train, predictors):
    print("running grid search for parameters")
    # Grid seach on subsample and max_features
    parameters_grid = {
        #'max_depth': range(3, 10, 2),
        #'min_child_weight': range(1, 6, 2),
        #'gamma': [i / 10.0 for i in range(0, 5)],
        # other parameters
        #'subsample': [i / 10.0 for i in range(5, 10)],
        #'colsample_bytree': [i / 10.0 for i in range(5, 10)],
        'reg_alpha': [0, 0.001, 0.005, 0.01, 0.05]
    }

    grid_search = GridSearchCV(
        estimator=model, param_grid=parameters_grid, cv=4, verbose=2)
    grid_search.fit(df_train[predictors], df_train['target'])
    print("%s" % grid_search.cv_results_)
    print("%s, %s" % (grid_search.best_params_, grid_search.best_score_))
    return grid_search


# Read in our input data
train_df = pd.read_csv('../input/train.csv', na_values="-1")
test_df = pd.read_csv('../input/test.csv', na_values="-1")

train_df.corr().to_csv("train_corr.csv")
test_df.corr().to_csv("test_corr.csv")

# train_df = train_df.fillna(train_df.mean())
# test_df = test_df.fillna(test_df.mean())

y_train = train_df['target'].values

x_train = train_df.drop(['target', 'id'], axis=1)
x_test = test_df.drop(['id'], axis=1)

# drop ps_calc_ columns
col_to_drop = train_df.columns[train_df.columns.str.startswith('ps_calc_')]
train_df = train_df.drop(col_to_drop, axis=1)
test_df = test_df.drop(col_to_drop, axis=1)
# drop ps_ind_11_bin, ps_ind_13_bin, and ps_ind_12_bin
train_df = train_df.drop(['ps_ind_11_bin', 'ps_ind_12_bin', 'ps_ind_13_bin'], axis=1)
test_df = test_df.drop(['ps_ind_11_bin', 'ps_ind_12_bin', 'ps_ind_13_bin'], axis=1)

# Take a random 20% of the dataset as validation data
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.3, random_state=1981)

gc.collect()
print("loaded data")

parameters = {
    'learning_rate': 0.02,
    'max_depth': 4,
    'min_child_weight': 1,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'objective': 'binary:logistic',
    'scale_pos_weight': 1,
    'random_state': 1987,
    'silent': True,
}

model = XGBClassifier()
model.set_params(**parameters)

# single_run(parameters, d_train, d_valid, d_test)

predictors = [x for x in train_df.columns if x not in ['target', 'id']]

# multiple_run(model, train_df, predictors)
grid_search(model, train_df, predictors)
