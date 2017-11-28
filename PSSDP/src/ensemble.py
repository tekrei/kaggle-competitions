import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score


class Ensemble(object):
    def __init__(self, n_splits, base_models, stacker=LogisticRegression()):
        self.n_splits = n_splits
        self.base_models = base_models
        self.stacker = stacker

    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)

        folds = list(StratifiedKFold(
            n_splits=self.n_splits, shuffle=True).split(X, y))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))
        for i, clf in enumerate(self.base_models):

            S_test_i = np.zeros((T.shape[0], self.n_splits))

            for j, (train_idx, test_idx) in enumerate(folds):
                start = time.time()
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]

                print("Fit %s fold %d" % (str(clf).split('(')[0], j + 1))
                clf.fit(X_train, y_train)
                cross_score = cross_val_score(clf, X_train, y_train, cv=self.n_splits, scoring='roc_auc')
                print("\tcross_score: %.4f" % cross_score.mean())
                print("\ttraining in %.2f seconds" % (time.time() - start))
                start = time.time()
                y_pred = clf.predict_proba(X_holdout)[:, 1]
                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict_proba(T)[:, 1]
                print("\tpredicted in %.2f seconds" % (time.time() - start))
            S_test[:, i] = S_test_i.mean(axis=1)
        start = time.time()
        results = cross_val_score(
            self.stacker, S_train, y, cv=self.n_splits, scoring='roc_auc')
        print("Stacker score: %.4f (%.2f seconds)" %
              (results.mean(), time.time() - start))
        print("GINI ~=%.4f"%(2 * results.mean() - 1))
        start = time.time()
        self.stacker.fit(S_train, y)
        res = self.stacker.predict_proba(S_test)[:, 1]
        print("\tpredicted in %.2f seconds" % (time.time() - start))
        return res
