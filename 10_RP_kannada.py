import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np
from numpy import genfromtxt
from sklearn import random_projection
from sklearn.preprocessing import StandardScaler

start = time.time()
my_data = genfromtxt('datasets/kannada_letters.csv', delimiter=',', dtype=str)
end = time.time()
print('Dataset read in', end - start, 'seconds')
X = my_data[1:60001, 1:785].astype(float)
y = my_data[1:60001, 0].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(np.shape(X_train))
print(np.shape(X_test))

scaler = StandardScaler()
# Fit on training set only.
scaler.fit(X_train)
# Apply transform to both the training set and the test set.
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

k = 25
transformer_gauss = random_projection.GaussianRandomProjection(n_components=k)
transformer_sparse = random_projection.SparseRandomProjection(n_components=k)
X_train_gauss = transformer_gauss.fit_transform(X_train)
X_train_sparse = transformer_sparse.fit_transform(X_train)
X_test_gauss = transformer_gauss.fit_transform(X_test)
X_test_sparse = transformer_sparse.fit_transform(X_test)

print(np.shape(X_train_gauss))
print(np.shape(X_test_gauss))

print(k)
start = time.time()
clf = SVC(kernel='linear')
print('fitting')
clf.fit(X_train_sparse, y_train)
print('calculating train accuracy')
train_accuracy = clf.score(X_train_sparse, y_train)
print('calculating test accuracy')
test_accuracy = clf.score(X_test_sparse, y_test)
print(train_accuracy)
print(test_accuracy)
end = time.time()
print('Time elapsed :', end - start, 'seconds')
