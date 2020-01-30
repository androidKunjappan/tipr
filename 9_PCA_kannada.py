import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np
from numpy import genfromtxt
from sklearn.metrics import classification_report
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


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

# Make an instance of the Model
pca = PCA(120)
pca.fit(X_train)

X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

print(np.shape(X_train))
print(np.shape(X_test))

for k in [120, 100, 30, 60]:
    print(k)
    start = time.time()
    clf = SVC(kernel='linear')
    clf.fit(X_train[:,0:k], y_train)
    train_accuracy = clf.score(X_train[:,0:k], y_train)
    test_accuracy = clf.score(X_test[:,0:k], y_test)
    print(train_accuracy)
    print(test_accuracy)
    end = time.time()
    print('Time elapsed :', end - start, 'seconds')

#
# # Build full model with selected features
# clf.fit(X_train[:, feat_cols], y_train)
# train_accuracy = clf.score(X_train[:, feat_cols], y_train)
# test_accuracy = clf.score(X_test[:, feat_cols], y_test)
#
# y_train_pred = clf.predict(X_train[:, feat_cols])
# y_test_pred = clf.predict(X_test[:, feat_cols])
# print("Train Accuracy:", train_accuracy)
# print("Test Accuracy:", test_accuracy)
#
# print("============ Training Classification Report ============")
# print(classification_report(y_train, y_train_pred))
# print('\n')
#
# print("============ Testing Classification Report ============")
# print(classification_report(y_test, y_test_pred))
# print('\n')
