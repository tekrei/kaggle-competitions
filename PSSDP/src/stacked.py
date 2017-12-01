# The source code is based on https://www.kaggle.com/yekenot/simple-stacker-lb-0-284

import gc
import time

import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier

from ensemble import Ensemble
from utility import read_data, preprocess

start = time.time()

# Read in our input data
df_train, df_test = read_data()

df_train, df_test = preprocess(df_train, df_test)

id_test = df_test['id'].values

X = df_train.drop(['id', 'target'], axis=1)
X_test = df_test[X.columns].values
X = X.values
y = df_train['target'].values

df_train = None
df_test = None

print("Loaded and prepared data in %.2f seconds" % (time.time() - start))

gc.collect()

start = time.time()

# LightGBM params
lgb_params = {
    'max_depth': 6,
    'num_leaves': 2^6+1,
    'learning_rate': 0.1,
    'n_estimators': 450,
    'max_bin': 10,
    'subsample':0.8,
    'subsample_freq':10,
    'colsample_bytree':0.8,
    'min_child_samples':500
}

lgb_params2 = {
    'max_depth': 6,
    'num_leaves': 2^6+1,
    'learning_rate': 0.05,
    'n_estimators': 550,
    'max_bin': 10,
    'subsample':0.8,
    'subsample_freq':10,
    'colsample_bytree':0.8,
    'min_child_samples':500
}

lgb_params3 = {
    'max_depth': 4,
    'num_leaves': 2^4+1,
    'learning_rate': 0.1,
    'n_estimators': 650,
    'max_bin': 10,
    'subsample':0.8,
    'subsample_freq':10,
    'colsample_bytree':0.8,
    'min_child_samples':500
}

# RandomForest params
rf_params = {
    'n_estimators': 1000,
    'max_depth': 5,
    'min_samples_split': 70,
    'min_samples_leaf': 30,
}


# ExtraTrees params
et_params = {
    'n_estimators': 155,
    'max_features': 0.3,
    'max_depth': 6,
    'min_samples_split': 40,
    'min_samples_leaf': 18,
}


# XGBoost params
xgb_params = {
    'objective':'binary:logistic',
    'learning_rate': 0.04,
    'n_estimators': 200,
    'num_boost_round': 1000,
    'max_depth': 5,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'min_child_weight':10,
	'tree_method':'hist' # must be faster
}

xgb_params2 = {
    'objective':'binary:logistic',
    'learning_rate': 0.08,
    'n_estimators': 200,
    'num_boost_round': 1000,
    'max_depth': 5,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'min_child_weight':10,
	'tree_method':'hist' # must be faster
}


# CatBoost params
cat_params = {
    'iterations': 1000,
    'depth': 8,
    'rsm': 0.95,
    'learning_rate': 0.03,
    'l2_leaf_reg': 3.5,
    'border_count': 8,
    'gradient_iterations': 4
}

# Gradient Boosting params
gb_params = {
    'max_depth': 6,
    'n_estimators': 1000,
    'learning_rate': 0.025,
    'subsample': 0.9
}

models = {
    "LGB-1": LGBMClassifier(**lgb_params),
    "XGB-1": XGBClassifier(**xgb_params),
    "LGB-2": LGBMClassifier(**lgb_params2),
    #"LGB-3": LGBMClassifier(**lgb_params3),
    "XGB-2": XGBClassifier(**xgb_params2),
    #"CAT": CatBoostClassifier(**cat_params),
    #"GBM": GradientBoostingClassifier(**gb_params),
    #"RF": RandomForestClassifier(**rf_params),
    #"ET": ExtraTreesClassifier(**et_params),
    #"ABC": AdaBoostClassifier(n_estimators=100),
}

start = time.time()
stack = Ensemble(4, models.values(), stacker=SGDClassifier(loss="log", max_iter=1000))
y_pred = stack.fit_predict(X, y, X_test)
print("Finished ensembling in %.2f seconds" % (time.time() - start))

sub = pd.DataFrame()
sub['id'] = id_test
sub['target'] = y_pred
sub.to_csv("%s.csv" % ("-".join(models.keys())), index=False)
