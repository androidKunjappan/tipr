# importing required libraries
import numpy as np
from numpy import genfromtxt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


# reading csv file and extracting class column to y. 
my_data = genfromtxt('datasets/pd_speech.csv', delimiter=',', dtype=str)
# Create feature and target arrays
X = my_data[2:758, 1:754].astype(float)
y = my_data[2:758, 754].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

#print(X)
#print(y)
features = np.arange(1, 30)
train_accuracy = np.empty(len(features))
test_accuracy = np.empty(len(features))

# Loop over K values
for i, k in enumerate(features):
    print(i, k)
    clf = SVC(kernel='linear')
    clf.fit(X_train[:,0:k], y_train)
    train_accuracy[i] = clf.score(X_train[:,0:k], y_train)
    test_accuracy[i] = clf.score(X_test[:,0:k], y_test)
    print(train_accuracy[i])
    print(test_accuracy[i])

# Generate plot
plt.plot(features, test_accuracy, label='Testing dataset Accuracy')
plt.plot(features, train_accuracy, label='Training dataset Accuracy')

plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.show()
