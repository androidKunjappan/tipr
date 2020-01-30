import sys
import read_datasets as rd
import run_classifier as rc
import run_dim_red as dr
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


def initialize():  # function to read input initialize matrix
    if len(sys.argv) != 3 and len(sys.argv) != 4:
        print("The number of arguments doesn't match 2 or 3..\nSee README section before running. Exiting!!")
        sys.exit()
    dataset = sys.argv[1]
    classifier = sys.argv[2]
    dim_red = ''
    if len(sys.argv) == 4:
        dim_red = sys.argv[3]
    if dataset not in ['iris', 'letter', 'pd_speech', 'kannada']:
        print('Invalid dataset received as argument')
        print('Please enter iris, letter, pd_speech or kannada')
        print('Exiting!!')
        sys.exit()
    if classifier not in ['knnc', 'svm', 'rf', 'xgboost', 'nn']:
        print('Invalid classifier received as argument')
        print('Please enter knnc, svm, rf, xgboost or nn')
        print('Exiting!!')
        sys.exit()
    if len(sys.argv) == 4:
        if dim_red not in ['sffs', 'mi', 'lsh', 'pca', 'rp']:
            print('Invalid classifier received as argument')
            print('Please enter sffs, mi, lsh, pca or rp')
            print('Exiting!!')
            sys.exit()
    return dataset, classifier, dim_red


def main():
    dataset, classifier, dim_red = initialize()

    # Read dataset
    if dataset == 'iris':
        X, y = rd.read_iris()
    elif dataset == 'letter':
        X, y = rd.read_letter()
    elif dataset == 'pd_speech':
        X, y = rd.read_pd_speech()
    elif dataset == 'kannada':
        X, y = rd.read_kannada()
    else:
        print('Unknown error!! Exiting!!')
        sys.exit()

    # Split the data inti train(80%) and test(20%) sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    if dim_red == 'sffs':
        normalize = 'yes'
        k_features = cv = neighbors = 0
        hidden_layer_sizes = (8, 8, 8)
        max_iter = 500
        if dataset in ['iris', 'letter']:
            normalize = 'no'
        if dataset == 'iris':
            k_features = 3
            cv = 5
            neighbors = 1
            max_iter = 1000
        elif dataset == 'letter':
            k_features = 10
            cv = 3
            neighbors = 4
        elif dataset == 'pd_speech':
            k_features = 20
            cv = 3
            neighbors = 1
        elif dataset == 'kannada':
            k_features = 15
            cv = 3
            neighbors = 5
            hidden_layer_sizes = (754, 150, 15)
        clf = RandomForestClassifier(n_estimators=10)
        if classifier == 'knnc':
            clf = KNeighborsClassifier(n_neighbors=neighbors, weights='distance')
        elif classifier == 'svd':
            clf = SVC(kernel='linear')
        elif classifier == 'rf':
            clf = RandomForestClassifier(n_estimators=10)
        elif classifier == 'nn':
            clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation='relu', solver='adam', max_iter=max_iter)
        elif classifier == 'xgboost':
            print('XGBoost unsupported with SFFS.. proceeding to run without dimensionality reduction')
        if classifier != 'xgboost':
            X_train, X_test = dr.run_sffs(X_train, X_test, y_train, y_test, clf, normalize, k_features, cv)

    # Run classifier
    if classifier == 'knnc':
        normalize = 'yes'
        if dataset in ['iris', 'letter']:
            normalize = 'no'
        max_neighbors = 15
        train_accuracy, test_accuracy, neighbor = rc.run_knnc(X_train, X_test, y_train, y_test, max_neighbors, normalize)
        print('*********Best Results********')
        print('No of neighbors =', neighbor)
        print('Training dataset accuracy\t:', train_accuracy)
        print('Testing dataset accuracy\t:', test_accuracy)
    elif classifier == 'svm':
        normalize = 'yes'
        if dataset in ['iris', 'letter', 'kannada']:
            normalize = 'no'
        train_accuracy, test_accuracy = rc.run_svm(X_train, X_test, y_train, y_test, normalize)
        print('*********Result********')
        print('Training dataset accuracy\t:', train_accuracy)
        print('Testing dataset accuracy\t:', test_accuracy)
    elif classifier == 'rf':
        normalize = 'yes'
        if dataset in ['iris', 'letter']:
            normalize = 'no'
        train_accuracy, test_accuracy = rc.run_rf(X_train, X_test, y_train, y_test, normalize)
        print('*********Result********')
        print('Training dataset accuracy\t:', train_accuracy)
        print('Testing dataset accuracy\t:', test_accuracy)
    elif classifier == 'xgboost':
        normalize = 'yes'
        max_depth = num_class = num_round = 0
        if dataset in ['iris', 'letter']:
            normalize = 'no'
            le = preprocessing.LabelEncoder()
            le.fit(y)
            y = le.transform(y)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)
        if dataset == 'iris':
            max_depth = 4
            num_class = 3
            num_round = 10
        elif dataset == 'letter':
            max_depth = 15
            num_class = 26
            num_round = 60
        elif dataset == 'pd_speech':
            max_depth = 6
            num_class = 2
            num_round = 25
        elif dataset == 'kannada':
            max_depth = 10
            num_class = 10
            num_round = 20
        train_accuracy, test_accuracy = rc.run_xgboost(X_train, X_test, y_train, y_test, normalize, max_depth, num_class, num_round)
        print('*********Result********')
        print('Training dataset accuracy\t:', train_accuracy)
        print('Testing dataset accuracy\t:', test_accuracy)
    elif classifier == 'nn':
        normalize = 'yes'
        max_iter = 500
        hidden_layer_sizes = (8, 8, 8)
        if dataset in ['iris', 'letter']:
            normalize = 'no'
        if dataset == 'iris':
            max_iter = 1000
        elif dataset == 'kannada':
            hidden_layer_sizes=(754, 150, 15)
        train_accuracy, test_accuracy = rc.run_nn(X_train, X_test, y_train, y_test, normalize, max_iter, hidden_layer_sizes)
        print('*********Result********')
        print('Training dataset accuracy\t:', train_accuracy)
        print('Testing dataset accuracy\t:', test_accuracy)


main()
