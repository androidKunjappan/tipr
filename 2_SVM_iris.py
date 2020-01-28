# importing required libraries
import numpy as np
from numpy import genfromtxt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import train_test_split


my_data = genfromtxt('datasets/Iris.csv', delimiter=',', dtype=str)
X = my_data[1:151, 1:5].astype(float)
y = my_data[1:151, 5].astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

clf = SVC(kernel='linear')
clf.fit(X_train, y_train)
train_accuracy = clf.score(X_train, y_train)
test_accuracy = clf.score(X_test, y_test)

y_pred = clf.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Train Accuracy:", test_accuracy)
print("Test Accuracy:", test_accuracy)
