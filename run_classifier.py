import time
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score
from sklearn.neural_network import MLPClassifier


def normalize_features(X_train, X_test):
    print('Normalizing features..')
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


def run_knnc(X_train, X_test, y_train, y_test, max_neighbors, normalize):
    if normalize == 'yes':
        X_train, X_test = normalize_features(X_train, X_test)
    train_accuracy = np.empty(max_neighbors)
    test_accuracy = np.empty(max_neighbors)
    neighbors = np.arange(1, max_neighbors + 1)
    start = time.time()
    for k in range(max_neighbors):
        print('Running for', k + 1, 'neighbors..')
        knn = KNeighborsClassifier(n_neighbors=k + 1, weights='distance')
        knn.fit(X_train, y_train)

        train_accuracy[k] = knn.score(X_train, y_train)
        test_accuracy[k] = knn.score(X_test, y_test)
        print('Training dataset accuracy\t:', train_accuracy[k])
        print('Testing dataset accuracy\t:', test_accuracy[k])
    end = time.time()
    print('KNNC classification done in', end - start, 'seconds\n')
    plt.plot(neighbors, test_accuracy, label='Testing dataset Accuracy')
    plt.plot(neighbors, train_accuracy, label='Training dataset Accuracy')
    plt.legend()
    plt.xlabel('n_neighbors')
    plt.ylabel('Accuracy')
    plt.show()

    max_accuracy = 0
    best_index = 0
    for k in range(max_neighbors):
        if test_accuracy[k] > max_accuracy:
            max_accuracy = test_accuracy[k]
            best_index = k
    return train_accuracy[best_index], test_accuracy[best_index], best_index + 1


def run_svm(X_train, X_test, y_train, y_test, normalize='no'):
    if normalize == 'yes':
        X_train, X_test = normalize_features(X_train, X_test)
    print('Starting SVM classification..')
    start = time.time()
    clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)
    train_accuracy = clf.score(X_train, y_train)
    test_accuracy = clf.score(X_test, y_test)
    end = time.time()
    print('SVM classification done in', end - start, 'seconds\n')
    return train_accuracy, test_accuracy


def run_rf(X_train, X_test, y_train, y_test, normalize):
    if normalize == 'yes':
        X_train, X_test = normalize_features(X_train, X_test)
    print('Starting Random Forest classification..')
    start = time.time()
    rfc = RandomForestClassifier(n_estimators=10)
    rfc.fit(X_train, y_train)
    train_accuracy = rfc.score(X_train, y_train)
    test_accuracy = rfc.score(X_test, y_test)
    end = time.time()
    print('Random Forest classification done in', end - start, 'seconds\n')
    return train_accuracy, test_accuracy


def run_xgboost(X_train, X_test, y_train, y_test, normalize, max_depth, num_class, num_round):
    if normalize == 'yes':
        X_train, X_test = normalize_features(X_train, X_test)
    print('Starting XGBoost classification..')
    start = time.time()
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    param = {
        'max_depth': max_depth,  # the maximum depth of each tree
        'eta': 0.3,  # the training step for each iteration
        'silent': 1,  # logging mode - quiet
        'objective': 'multi:softprob',  # error evaluation for multiclass training
        'num_class': num_class}  # the number of classes that exist in this datset

    bst = xgb.train(param, dtrain, num_round)
    train_preds = bst.predict(dtrain)
    test_preds = bst.predict(dtest)

    # extracting most confident predictions
    train_best_preds = np.asarray([np.argmax(line) for line in train_preds])
    test_best_preds = np.asarray([np.argmax(line) for line in test_preds])
    train_accuracy = precision_score(y_train, train_best_preds, average='macro')
    test_accuracy = precision_score(y_test, test_best_preds, average='macro')
    end = time.time()
    print('XGBoost classification done in', end - start, 'seconds\n')
    return train_accuracy, test_accuracy


def run_nn(X_train, X_test, y_train, y_test, normalize, max_iter, hidden_layer_sizes):
    if normalize == 'yes':
        X_train, X_test = normalize_features(X_train, X_test)
    print('Starting Neural Network classification..')
    start = time.time()
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation='relu', solver='adam', max_iter=max_iter)
    mlp.fit(X_train, y_train)
    train_accuracy = mlp.score(X_train, y_train)
    test_accuracy = mlp.score(X_test, y_test)
    end = time.time()
    print('Neural Network classification done in', end - start, 'seconds\n')
    return train_accuracy, test_accuracy
