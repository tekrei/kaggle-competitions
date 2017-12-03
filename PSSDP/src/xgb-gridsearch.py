from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV

from utility import gini_scorer, read_data

# grid search example using XGBoost

def grid_search(model, x_train, y_train):
    print("running grid search for parameters")
    parameters_grid = {
        'max_depth': [4, 5, 6],
        'gamma': [0, 0.25, 0.5, 0.75, 1],
        'subsample': [0.7, 0.8, 0.9, 1],
        'colsample_bytree': [0.7, 0.8, 0.9, 1],
        #'reg_alpha': [0, 0.001, 0.005, 0.01, 0.05]
    }

    grid_search = GridSearchCV(estimator=model, param_grid=parameters_grid,
                               cv=4, verbose=3, n_jobs=4, scoring=gini_scorer)
    grid_search.fit(x_train, y_train)
    print("%s" % grid_search.cv_results_)
    print("%s, %s" % (grid_search.best_params_, grid_search.best_score_))
    return grid_search


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

df_train = df_train.drop(drop_set, axis=1)
df_test = df_test.drop(drop_set, axis=1)


model = XGBClassifier(learning_rate=0.1, max_depth=4, n_estimators=100,
                      subsample=0.8, colsample_bytree=0.8,
                      objective='binary:logistic', random_state=1981)
print("created model")

grid_search(model, df_train, df_train['target'].values)
