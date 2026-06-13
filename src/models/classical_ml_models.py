# -*- coding: utf-8 -*-
"""Classical ML models + Optuna search spaces.

Library module imported by training/run_single_experiment.py. Exposes the four
estimators used in the paper (KNN, SVM, RandomForest, XGBoost) and their Optuna
hyper-parameter spaces. The spaces include extra KNN
`algorithm`/`leaf_size`; SVM poly/sigmoid kernels with `degree`/`coef0`; RF
`bootstrap`/`class_weight`; XGB `min_child_weight`/`gamma` and wider depth.

Selection metric and CV scheme live in the training driver: 200 Optuna TPE
trials, no pruner (NopPruner), selection by mean weighted-F1 over the
StratifiedGroupKFold(5) folds. Normalization/StandardScaler are fit per fold
(see normalization.spectral_normalization.normalize_train_test) to avoid leakage.
"""
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

ML_MODELS = ['knn', 'svm', 'rf', 'xgb']


def build_ml(model, params, n_classes, seed=42, proba=False):
    """Instantiate an estimator. `proba` only matters for SVM (probability=...)."""
    if model == 'knn':
        return KNeighborsClassifier(**params)
    if model == 'svm':
        return SVC(probability=proba, random_state=seed, **params)
    if model == 'rf':
        return RandomForestClassifier(random_state=seed, n_jobs=-1, **params)
    if model == 'xgb':
        return XGBClassifier(objective='multi:softprob', num_class=n_classes,
                             eval_metric='mlogloss', random_state=seed, n_jobs=-1,
                             tree_method='hist',
                             device=os.environ.get('XGB_DEVICE', 'cpu'), **params)
    raise ValueError(model)


def suggest_ml(model, t):
    """Optuna search space for one estimator."""
    if model == 'knn':
        return dict(n_neighbors=t.suggest_int('n_neighbors', 1, 50),
                    weights=t.suggest_categorical('weights', ['uniform', 'distance']),
                    p=t.suggest_int('p', 1, 3),
                    algorithm=t.suggest_categorical('algorithm',
                                                    ['auto', 'ball_tree', 'kd_tree', 'brute']),
                    leaf_size=t.suggest_int('leaf_size', 10, 50))
    if model == 'svm':
        k = t.suggest_categorical('kernel', ['rbf', 'linear', 'poly', 'sigmoid'])
        d = dict(kernel=k, C=t.suggest_float('C', 1e-2, 1e2, log=True),
                 gamma=t.suggest_float('gamma', 1e-4, 1e1, log=True),
                 class_weight=t.suggest_categorical('class_weight', [None, 'balanced']))
        if k == 'poly':
            d['degree'] = t.suggest_int('degree', 2, 5)
        if k in ('poly', 'sigmoid'):
            d['coef0'] = t.suggest_float('coef0', 0.0, 1.0)
        return d
    if model == 'rf':
        return dict(n_estimators=t.suggest_int('n_estimators', 50, 500),
                    max_depth=t.suggest_int('max_depth', 3, 30),
                    min_samples_split=t.suggest_int('min_samples_split', 2, 10),
                    min_samples_leaf=t.suggest_int('min_samples_leaf', 1, 10),
                    max_features=t.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                    bootstrap=t.suggest_categorical('bootstrap', [True, False]),
                    class_weight=t.suggest_categorical(
                        'class_weight', [None, 'balanced', 'balanced_subsample']))
    if model == 'xgb':
        return dict(max_depth=t.suggest_int('max_depth', 3, 15),
                    learning_rate=t.suggest_float('learning_rate', 1e-3, 0.5, log=True),
                    n_estimators=t.suggest_int('n_estimators', 50, 500),
                    subsample=t.suggest_float('subsample', 0.5, 1.0),
                    colsample_bytree=t.suggest_float('colsample_bytree', 0.5, 1.0),
                    min_child_weight=t.suggest_float('min_child_weight', 1, 10),
                    gamma=t.suggest_float('gamma', 0.0, 0.5),
                    reg_alpha=t.suggest_float('reg_alpha', 1e-3, 10, log=True),
                    reg_lambda=t.suggest_float('reg_lambda', 1e-3, 10, log=True))
    raise ValueError(model)
