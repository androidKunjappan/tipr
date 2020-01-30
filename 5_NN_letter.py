from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from numpy import genfromtxt
from sklearn.neural_network import MLPClassifier

my_data = genfromtxt('datasets/letter-recognition.csv', delimiter=',', dtype=str)

# Create feature and target arrays
X = my_data[0:20000, 1:17].astype(int)
y = my_data[0:20000, 0].astype(str)

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

mlp = MLPClassifier(hidden_layer_sizes=(19, ), activation='relu', solver='adam', max_iter=1000)
mlp.fit(X_train, y_train)

predict_train = mlp.predict(X_train)
predict_test = mlp.predict(X_test)

print(mlp.score(X_train, y_train))
print(mlp.score(X_test, y_test))

print("============ Classification Report ============")
print(classification_report(y_test, predict_test))
print('\n')
