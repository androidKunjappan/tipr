# importing required libraries
import numpy as np
from numpy import genfromtxt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import train_test_split


# reading csv file and extracting class column to y. 
my_data = genfromtxt('datasets/letter-recognition.csv', delimiter=',', dtype=str)
# Create feature and target arrays
X = my_data[0:20000, 1:17].astype(int)
y = my_data[0:20000, 0].astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

clf = SVC(kernel='linear')
clf.fit(X_train, y_train)
train_accuracy = clf.score(X_train, y_train)
test_accuracy = clf.score(X_test, y_test)

y_pred = clf.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Train Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)
