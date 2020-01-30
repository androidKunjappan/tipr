import time
from numpy import genfromtxt


def read_iris():
    print('Reading Iris dataset..')
    my_data = genfromtxt('datasets/Iris.csv', delimiter=',', dtype=str)
    X = my_data[1:151, 1:5].astype(float)
    y = my_data[1:151, 5].astype(str)
    return X, y


def read_letter():
    print('Reading Letter Recognition dataset..')
    my_data = genfromtxt('datasets/letter-recognition.csv', delimiter=',', dtype=str)
    X = my_data[0:20000, 1:17].astype(int)
    y = my_data[0:20000, 0].astype(str)
    return X, y


def read_pd_speech():
    print('Reading Parkinson\'s disease speech dataset..')
    my_data = genfromtxt('datasets/pd_speech.csv', delimiter=',', dtype=str)
    X = my_data[2:758, 1:754].astype(float)
    y = my_data[2:758, 754].astype(int)
    return X, y


def read_kannada():
    print('Reading Kannada dataset..')
    start = time.time()
    my_data = genfromtxt('datasets/kannada_letters.csv', delimiter=',', dtype=str)
    X = my_data[1:60001, 1:785].astype(float)
    y = my_data[1:60001, 0].astype(int)
    end = time.time()
    print('Dataset read in', end - start, 'seconds')
    return X, y