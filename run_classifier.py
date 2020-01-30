import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def normalize_features(X_train, X_test):
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
    neighbors = np.arange(1, max_neighbors+1)
    start = time.time()
    for k in range(max_neighbors):
        print('Running for', k+1, 'neighbors..')
        knn = KNeighborsClassifier(n_neighbors=k+1, weights='distance')
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
    return train_accuracy[best_index], test_accuracy[best_index], best_index+1