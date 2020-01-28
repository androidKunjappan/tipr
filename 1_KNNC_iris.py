# Import necessary modules 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt

#irisData = load_iris()

my_data = genfromtxt('datasets/Iris.csv', delimiter=',', dtype=None)
X = my_data[1:151, 1:5].astype(float)
y = my_data[1:151, 5].astype(str)

# Split into training and test set
#X_train, X_test, y_train, y_test = train_test_split(
#    X, y, test_size=0.2, random_state=30)

X_train = X
y_train = y
neighbors = np.arange(1, 151)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over K values 
for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    knn.fit(X_train, y_train)

    # Compute traning and test data accuracy
    train_accuracy[i] = knn.score(X_train, y_train)
    #test_accuracy[i] = knn.score(X_test, y_test)

# Generate plot 
#plt.plot(neighbors, test_accuracy, label='Testing dataset Accuracy')
plt.plot(neighbors, train_accuracy, label='Training dataset Accuracy')

plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.show()
