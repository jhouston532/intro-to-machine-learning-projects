# Imports
import io
import urllib.request

import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def iterations(X, y, i):
    test_acc_arr, train_acc_arr = [], []

    for h in range(i):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=h
        )
        # this enforces the 80-20 split, with different random states
        # random states are used so that results are replicable, but also different

        knn = KNeighborsClassifier(n_neighbors=3, weights="uniform")
        # uniform weights does the whole "no distance weighting"

        knn.fit(X_train, y_train)  # and no data cleanup

        # capture the test and train scores
        test_acc = knn.score(X_test, y_test)
        train_acc = knn.score(X_train, y_train)
        test_acc_arr.append(test_acc)
        train_acc_arr.append(train_acc)

    # loop end
    pd.DataFrame(
        {"Train Acc": train_acc_arr, "Test Acc": test_acc_arr}, index=range(i)
    ).plot(
        marker="o",
        xlabel="Random State",
        ylabel="Accuracy",
        title="KNN Accuracy vs Random State",
        grid=True,
    )
    plt.show()

    # print scores
    print("Test set accuracy scores:\t\t", test_acc_arr)
    print("Train set accuracy scores:\t\t", train_acc_arr)
    train_avg = sum(train_acc_arr) / len(train_acc_arr)
    test_avg = sum(test_acc_arr) / len(test_acc_arr)

    print("Average Training Accuracy:\t\t", train_avg)
    print("Average Testing Accuracy:\t\t", test_avg)


def main():
    #
    # Get the data
    url = "https://raw.githubusercontent.com/BYU-CS-472/CS472/master/datasets/magic_telescope_train.arff"
    with urllib.request.urlopen(url) as response:
        raw_data = response.read().decode("utf-8")

    # Put data into a dataframe
    data, _ = arff.loadarff(io.StringIO(raw_data))
    df = pd.DataFrame(data)

    # define X, y
    # X = the type column
    X = df.drop(columns=["class"])
    # y = everything bar X
    y = df["class"].str.decode("utf-8")

    
    
    # loop to capture results
    print("k=3, uniform weights, no normalization")
    iterations(X, y, 15)


if __name__ == "__main__":
    main()
