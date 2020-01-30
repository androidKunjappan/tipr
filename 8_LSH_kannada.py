from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from numpy import genfromtxt
from sklearn.preprocessing import StandardScaler

from LSH_ANN import LSH_ANN

# reading csv file and extracting class column to y.
my_data = genfromtxt('datasets/letter-recognition.csv', delimiter=',', dtype=str)
# Create feature and target arrays
X = my_data[0:20000, 1:17].astype(int)
y = my_data[0:20000, 0].astype(str)

my_data = genfromtxt('datasets/kannada_letters.csv', delimiter=',', dtype=str)

# Create feature and target arrays
X = my_data[1:60001, 1:785].astype(float)
y = my_data[1:60001, 0].astype(int)

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


for L in [20, 40, 60,80, 100]:
    for w in [10, 15, 25, 50, 100]:
        classifier = LSH_ANN(L, 15, w)
        classifier.train(X_train, y_train)
        #classifier.tune(X_test, y_test, range(1, 500, 20))
        #print(y_test)
        accuracy = classifier.test(X_test, y_test)
        print(
        "Attained an accuracy of: " + str(accuracy) + " for a number of neighbors = " + str(
            L) + " and a bucket width of "+ str(w) + ".")
