import time
from sklearn import random_projection
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler
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


def run_mi(X_train, X_test, y_train, y_test, normalize, k_features):
    if normalize == 'yes':
        X_train, X_test = normalize_features(X_train, X_test)

    print('Starting MI Dimensionality Reduction ..')
    start = time.time()
    fs = SelectKBest(score_func=mutual_info_classif, k=k_features)
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)
    end = time.time()
    print('\nMI done in', end - start, 'seconds\n')

    return X_train_fs, X_test_fs


def run_pca(X_train, X_test, y_train, y_test, normalize, k_features):
    if normalize == 'yes':
        X_train, X_test = normalize_features(X_train, X_test)

    print('Starting PCA Dimensionality Reduction ..')
    start = time.time()
    pca = PCA(k_features)
    pca.fit(X_train)
    X_train_fs = pca.transform(X_train)
    X_test_fs = pca.transform(X_test)
    end = time.time()
    print('\nPCA done in', end - start, 'seconds\n')
    return X_train_fs, X_test_fs


def run_rp(X_train, X_test, y_train, y_test, normalize, k_features):
    if normalize == 'yes':
        X_train, X_test = normalize_features(X_train, X_test)

    print('Starting RP Dimensionality Reduction ..')
    start = time.time()
    transformer_gauss = random_projection.GaussianRandomProjection(n_components=k_features)
    X_train_fs = transformer_gauss.fit_transform(X_train)
    X_test_fs = transformer_gauss.fit_transform(X_test)
    end = time.time()
    print('\nRP done in', end - start, 'seconds\n')

    return X_train_fs, X_test_fs
