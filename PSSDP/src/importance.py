from collections import OrderedDict
from operator import itemgetter

import matplotlib.pylab as plt
from pandas import DataFrame, get_dummies
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, plot_importance, plot_tree

# importance analysis with XGBoost

from utility import gini, gini_xgb, plot_corr, read_data, change_datatype

df_train, df_test = read_data()

# statistical information of each column on data frames
file = open('description.txt', 'w')
for predictor in [x for x in df_train.columns if x not in ['target', 'id']]:
    file.write("%s\n-------------------------\n"%predictor)
    file.write("Training Data (%d):\n%s\n"%(df_train[predictor].nunique(), df_train[predictor].describe()))
    file.write("Testing Data (%d):\n%s\n"%(df_test[predictor].nunique(), df_test[predictor].describe()))
    if set(df_train[predictor].unique())==set(df_test[predictor].unique()):
        file.write("Could be categorical because training and testing unique values are equal\n")
    if df_train[predictor].nunique()<10:
        file.write("Training Values: %s\n"%df_train[predictor].unique())
    if df_test[predictor].nunique()<10:
        file.write("Testing Values: %s\n"%df_test[predictor].unique())
    file.write("-------------------------\n")
file.close()

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
ohe_columns = ['ps_ind_01', 'ps_ind_03', 'ps_ind_15', 'ps_reg_02', 'ps_car_04_cat', 'ps_car_06_cat', 'ps_car_15']
df_train = get_dummies(df_train, columns = ohe_columns)
df_test = get_dummies(df_test, columns = ohe_columns)


df_train = df_train.drop(drop_set, axis=1)
df_test = df_test.drop(drop_set, axis=1)


predictors = [x for x in df_train.columns if x not in ['target', 'id']]

# save correlation between features and target
df_train[predictors].corrwith(df_train['target']).to_csv("correlation.csv")

plot_corr(df_train.corr())
plt.savefig("train_corr.pdf")
plot_corr(df_test.corr())
plt.savefig("test_corr.pdf")

X_train, X_test, y_train, y_test = train_test_split(
    df_train[predictors], df_train['target'], test_size=0.25, random_state=4242)


params = {'feval': gini_xgb, 'maximize': True}

model = XGBClassifier(**params)


model.fit(X_train, y_train)

# make predictions for test data and evaluate
y_pred = model.predict(X_test)
print("AUC: %.5f" % auc(y_test, [round(value) for value in y_pred]))


# plot feature importances
fig, ax = plt.subplots(figsize=(20, 20))
plot_importance(model, ax=ax)
plt.savefig("importance.pdf")

# plot tree
fig, ax = plt.subplots(figsize=(40, 40))
plot_tree(model, ax=ax)
plt.savefig("tree.pdf")

thresholds = OrderedDict(
    sorted(dict(zip(predictors, model.feature_importances_)).items(), key=itemgetter(1), reverse=True))

DataFrame.from_dict(thresholds, orient="index").to_csv("features.csv")
# Fit model using each importance as a threshold
# for feature, threshold in thresholds.items():
#     # select features using threshold
#     selection = SelectFromModel(model, threshold=threshold, prefit=True)
#     select_X_train = selection.transform(X_train)
#     # train model
#     selection_model = XGBClassifier(**params)
#     selection_model.fit(select_X_train, y_train)
#     # eval model
#     select_X_test = selection.transform(X_test)
#     y_pred = selection_model.predict(select_X_test)
#     accuracy = accuracy_score(y_test, [round(value) for value in y_pred])
#     print("Threshold=%.3f, n=%d, Accuracy: %.3f" %
#           (threshold, select_X_train.shape[1], accuracy))
