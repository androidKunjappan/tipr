import time
# import numpy as np
# import matplotlib.pyplot as plt
# import xgboost as xgb
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import precision_score
# from sklearn.neural_network import MLPClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as sfs


def normalize_features(X_train, X_test):
    print('Normalizing features..')
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


def run_sffs(X_train, X_test, y_train, y_test, clf, normalize, k_features, cv):
    if normalize == 'yes':
        X_train, X_test = normalize_features(X_train, X_test)
    print('Starting SFFS Dimensionality Reduction ..')
    start = time.time()
    sfs1 = sfs(clf,
               k_features=k_features,
               forward=True,
               floating=True,
               verbose=2,
               scoring='accuracy',
               cv=cv,
               n_jobs=-1)
    sfs1 = sfs1.fit(X_train, y_train)

    feat_cols = list(sfs1.k_feature_idx_)
    end = time.time()
    print('\nSFFS done in', end - start, 'seconds\n')
    return X_train[:, feat_cols], X_test[:, feat_cols]
