import sys
import read_datasets as rd
import run_classifier as rc
from sklearn.model_selection import train_test_split


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


main()
