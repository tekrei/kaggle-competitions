import gc

import xgboost as xgb
from sklearn.model_selection import StratifiedKFold

from utility import gini_xgb, preprocess, read_data

# Read in our input data
df_train, df_test = read_data()

df_train, df_test = preprocess(df_train, df_test)

# prepare xgboost parameters
params = {
    'min_child_weight': 10.0,
    'objective': 'binary:logistic',
    'max_depth': 7,
    'max_delta_step': 1.8,
    'colsample_bytree': 0.4,
    'subsample': 0.8,
    'eta': 0.025,
    'gamma': 0.65,
    'num_boost_round': 700,
    'silent': True
}


X = df_train.drop(['id', 'target'], axis=1)
features = X.columns
X = X.values
y = df_train['target'].values
submission = df_test['id'].to_frame()
submission['target'] = 0
df_train = None

X_test = df_test[features].values
df_test = None

gc.collect()

kfold = 4
skf = StratifiedKFold(n_splits=kfold, shuffle=True)
for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    print('fold %d of %d' % (i + 1, kfold))
    # prepare training data
    X_train, X_valid = X[train_index], X[test_index]
    y_train, y_valid = y[train_index], y[test_index]
    d_train = xgb.DMatrix(X_train, y_train)
    watchlist = [(d_train, 'train'), (xgb.DMatrix(X_valid, y_valid), 'valid')]
    # train the model
    model = xgb.train(params, d_train, 2000, watchlist,
                      early_stopping_rounds=100, feval=gini_xgb, maximize=True, verbose_eval=50)
    # predict using best iteration and save the result
    submission['target'] += model.predict(xgb.DMatrix(
        X_test), ntree_limit=model.best_ntree_limit) / kfold
    gc.collect()


# Create a submission file
submission.to_csv('stratified-%sfold.csv' % kfold, index=False)
