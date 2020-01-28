# Import necessary modules 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import precision_score
from sklearn import preprocessing
from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

#irisData = load_iris()

my_data = genfromtxt('datasets/pd_speech.csv', delimiter=',', dtype=str)
# Create feature and target arrays
X = my_data[2:758, 1:754].astype(float)
y = my_data[2:758, 754].astype(int)
# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=30)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

param = {
    'max_depth': 5,  # the maximum depth of each tree
    'eta': 0.3,  # the training step for each iteration
    'silent': 0,  # logging mode - quiet
    'objective': 'multi:softprob',  # error evaluation for multiclass training
    'num_class': 2}  # the number of classes that exist in this datset
num_round = 20  # the number of training iterations

#------------- numpy array ------------------
# training and testing - numpy matrices
bst = xgb.train(param, dtrain, num_round)
bst.dump_model('XGBoost_speech.txt')
preds = bst.predict(dtest)

# extracting most confident predictions
best_preds = np.asarray([np.argmax(line) for line in preds])
print("Numpy array precision:", precision_score(y_test, best_preds, average='macro'))
