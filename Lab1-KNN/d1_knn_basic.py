# Imports
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import numpy as np
import pandas as pd
from scipy.io import arff
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import urllib.request
import io
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Get the data
url = "https://raw.githubusercontent.com/BYU-CS-472/CS472/master/datasets/glass_train.arff"
with urllib.request.urlopen(url) as response:
  raw_data = response.read().decode("utf-8")

# Put data into a dataframe
data, metadata = arff.loadarff(io.StringIO(raw_data))
df = pd.DataFrame(data)

# Define X and Y
# X = the type column
X = df.drop(columns=["Type"])
# y = everything bar X
y = df["Type"].str.decode("utf-8")
rng = np.random.default_rng(42)

# different test/train sizes loop
# define these outside, so that they can be used outside
l1_test_acc, l1_train_acc = [], []

for i in range(25):
  # it said, a variety of sizes, so here you are
  test_size = rng.uniform(0.1, 0.4)

  # Split the data, with a randomized test set size
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=i)


  # Set N, the number of neighbors
  N = 5
  # create the models (because if y'all put both models in, I may as well see what's up)
  knn = KNeighborsClassifier(n_neighbors=N)
  knn.fit(X_train, y_train)

  y_pred = knn.predict(X_test)

  # let's see how they do
  print("Iteration ",i+1,"\nTest Size: ", test_size)
  test_acc = knn.score(X_test, y_test)
  train_acc = knn.score(X_train, y_train)
  l1_test_acc.append(test_acc)
  l1_train_acc.append(train_acc)
  # print("\tTest Acc so far:\t", l1_test_acc)
  # print("\tTrain Acc so far:\t", l1_train_acc)
  # print("\tOutput Probabilities:\t", knn.predict_proba(X_test))
  print("\tAccuracy Score:\t\t", accuracy_score(y_test, y_pred))
  print("\tRecall Score:\t\t", recall_score(y_test, y_pred, average=None, zero_division=0))
  print("\tPrecision Score:\t", precision_score(y_test, y_pred, average=None, zero_division=0))
  print("\tF1 Score:\t\t", f1_score(y_test, y_pred, average=None, zero_division=0), "\n\n")

# loop ends after 5 iterations
print("Final result knn")
print("Output Probabilities:\t", knn.predict_proba(X_test))
print("Test Acc:\t", l1_test_acc)
print("Train Acc:\t", l1_train_acc)

# different P loop
l2_test_acc, l2_train_acc = [], []

for i in range(15):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  knn = KNeighborsClassifier(n_neighbors=N, p=i+1)
  knn.fit(X_train, y_train)
  y_pred = knn.predict(X_test)
  print("Iteration p =",i+1,)

  test_acc = knn.score(X_test, y_test)
  train_acc = knn.score(X_train, y_train)
  l2_test_acc.append(test_acc)
  l2_train_acc.append(train_acc)
  # print("\tTest Acc so far:\t", l2_test_acc)
  # print("\tTrain Acc so far:\t", l2_train_acc)


  # copy and paste from above since we need the same stats basically
  print("\tAccuracy Score:\t\t", accuracy_score(y_test, y_pred))
  print("\tRecall Score:\t\t", recall_score(y_test, y_pred, average=None, zero_division=0))
  print("\tPrecision Score:\t", precision_score(y_test, y_pred, average=None, zero_division=0))
  print("\tF1 Score:\t\t", f1_score(y_test, y_pred, average=None, zero_division=0), "\n\n")


# end p loop
print("Test acc of P loop:\t\t", l2_test_acc)
print("Train acc of P loop:\t\t", l2_train_acc)