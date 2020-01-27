from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from numpy import genfromtxt

my_data = genfromtxt('datasets/kannada_letters.csv', delimiter=',', dtype=str)

# Create feature and target arrays
X = my_data[1:60001, 1:785].astype(float)
y = my_data[1:60001, 0].astype(int)

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

rfc_predict = rfc.predict(X_test)

print(rfc.score(X_train, y_train))
print(rfc.score(X_test, y_test))

print("============ Classification Report ============")
print(classification_report(y_test, rfc_predict))
print('\n')
