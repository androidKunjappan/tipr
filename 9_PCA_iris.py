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


my_data = genfromtxt('datasets/Iris.csv', delimiter=',', dtype=str)
X = my_data[1:151, 1:5].astype(float)
y = my_data[1:151, 5].astype(str)



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(np.shape(X_train))
print(np.shape(X_test))

# scaler = StandardScaler()
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)

# Make an instance of the Model
# full is best
pca = PCA(4)
pca.fit(X_train)

X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

print(np.shape(X_train))
print(np.shape(X_test))

k = np.shape(X_train)[1]
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
