import gc

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

from utility import gini_lgb, preprocess, read_data

# LightGBM params
params = {
#    'max_depth': 5,
    'learning_rate': 0.02,
    'n_estimators': 650,
    'max_bin': 10,
    'subsample':0.8,
    'subsample_freq':10,
    'colsample_bytree':0.8,
    'min_child_samples':500
}


# Read in our input data
df_train, df_test = read_data()

df_train, df_test = preprocess(df_train, df_test)

id_test = df_test['id'].values

X = df_train.drop(['id', 'target'], axis=1)
X_test = df_test[X.columns].values
X = X.values
y = df_train['target'].values

submission = df_test['id'].to_frame()
submission['target'] = 0

gc.collect()

kfold = 4
skf = StratifiedKFold(n_splits=kfold, shuffle=True)
for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    print('fold %d of %d' % (i + 1, kfold))
    # prepare training data
    X_train, X_valid = X[train_index], X[test_index]
    y_train, y_valid = y[train_index], y[test_index]
    # create dataset for lightgbm
    d_train = lgb.Dataset(X_train, y_train)
    d_eval = lgb.Dataset(X_valid, y_valid, reference=d_train)
    # train the model
    model = lgb.train(params, d_train, num_boost_round=2000,
                valid_sets=d_eval, early_stopping_rounds=100,
                feval=gini_lgb, verbose_eval=50)
    # predict using best iteration and save the result
    submission['target'] += model.predict(X_test, num_iteration=model.best_iteration) / kfold
    gc.collect()


# Create a submission file
submission.to_csv('stratified-%sfold-lgb.csv' % kfold, index=False)
