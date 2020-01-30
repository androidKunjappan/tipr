from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from numpy import genfromtxt

my_data = genfromtxt('datasets/pd_speech.csv', delimiter=',', dtype=str)

# Create feature and target arrays
X = my_data[2:758, 1:754].astype(float)
y = my_data[2:758, 754].astype(int)

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

fs = SelectKBest(score_func=mutual_info_classif, k=100)
fs.fit(X_train, y_train)
X_train_fs = fs.transform(X_train)
X_test_fs = fs.transform(X_test)

rfc = RandomForestClassifier()
rfc.fit(X_train_fs, y_train)

rfc_predict = rfc.predict(X_test_fs)

print(rfc.score(X_train_fs, y_train))
print(rfc.score(X_test_fs, y_test))

print("============ Classification Report ============")
print(classification_report(y_test, rfc_predict))
print('\n')
