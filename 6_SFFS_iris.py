from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from numpy import genfromtxt
from sklearn.metrics import classification_report
from mlxtend.feature_selection import SequentialFeatureSelector as sfs

my_data = genfromtxt('datasets/Iris.csv', delimiter=',', dtype=str)
X = my_data[1:151, 1:5].astype(float)
y = my_data[1:151, 5].astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

clf = SVC(kernel='linear')

sfs1 = sfs(clf,
           k_features=3,
           forward=True,
           floating=False,
           verbose=1,
           scoring='accuracy',
           cv=5)
sfs1 = sfs1.fit(X_train, y_train)

feat_cols = list(sfs1.k_feature_idx_)
print(feat_cols)

# Build full model with selected features
clf.fit(X_train[:, feat_cols], y_train)
train_accuracy = clf.score(X_train[:, feat_cols], y_train)
test_accuracy = clf.score(X_test[:, feat_cols], y_test)

y_train_pred = clf.predict(X_train[:, feat_cols])
y_test_pred = clf.predict(X_test[:, feat_cols])
print("Train Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)

print("============ Training Classification Report ============")
print(classification_report(y_train, y_train_pred))
print('\n')

print("============ Testing Classification Report ============")
print(classification_report(y_test, y_test_pred))
print('\n')
