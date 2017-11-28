import xgboost as xgb
import gc
from pandas import get_dummies

from utility import gini_xgb, read_data

# Read in our input data
df_train, df_test = read_data()

# drop ps_calc_ columns
col_to_drop = df_train.columns[df_train.columns.str.startswith('ps_calc_')]
df_train = df_train.drop(col_to_drop, axis=1)
df_test = df_test.drop(col_to_drop, axis=1)
# columns with very skewed data ['ps_ind_10_bin','ps_ind_11_bin','ps_ind_12_bin','ps_ind_13_bin'].
# columns with outliers ['ps_ind_14', 'ps_car_10_cat']
# drop low covariance columns
drop_set = ['ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin', 'ps_ind_13_bin', 'ps_ind_14',
            'ps_ind_18_bin', 'ps_car_10_cat', 'ps_car_11_cat']

# one hot encodable columns: 'ps_ind_01', 'ps_ind_03', 'ps_ind_15', 'ps_reg_02', 'ps_car_04_cat', 'ps_car_06_cat', 'ps_car_15'
# ohe_columns = ['ps_ind_01', 'ps_ind_03', 'ps_ind_15',
#               'ps_reg_02', 'ps_car_04_cat', 'ps_car_06_cat', 'ps_car_15']
# df_train = get_dummies(df_train, columns=ohe_columns)
# df_test = get_dummies(df_test, columns=ohe_columns)

df_train = df_train.drop(drop_set, axis=1)
df_test = df_test.drop(drop_set, axis=1)

# Set xgboost parameters
params = {'eta': 0.1, 'max_depth': 4, 'subsample': 0.8, 'colsample_bytree': 0.8, 'eval_metric': 'auc',
          'gamma': 1, 'reg_alpha': 0, 'reg_lambda': 1, 'objective': 'binary:logistic', 'silent': True}

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

# watchlist = [(d_train, 'train'), (xgb.DMatrix(X_valid, y_valid), 'valid')]
result = xgb.cv(params, xgb.DMatrix(X, y), 100, 8, early_stopping_rounds=10, feval=gini_xgb, maximize=True, verbose_eval=10)
print(result)
submission['target'] = model.predict(xgb.DMatrix(X_test))
gc.collect()


# Create a submission file
submission.to_csv('submission.csv', index=False)
