import io
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def training_size_check(
    X, y, min_size, max_size, interval=0.05, neighbors=5, rand_state_range=10
):
    # input data for a test set where the variable is the split
    # return the x axis and two data sets, test_sizes_arr train_acc and test_acc,

    # find how many intervals are between min and max
    n_steps = round((max_size - min_size) / interval) + 1

    # this calculates the x-axis that we're going to use to plot the graph
    test_sizes = np.round(np.linspace(min_size, max_size, n_steps), 4)

    train_acc_arr, test_acc_arr = [], []

    # for each test size in between the
    for test_size in test_sizes:
        train_scores, test_scores = [], []

        # check the test size over r different seeds
        for seed in range(rand_state_range):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=seed
            )

            # setup the knn
            knn = KNeighborsClassifier(n_neighbors=neighbors)
            knn.fit(X_train, y_train)

            # save the accuracies
            train_acc = knn.score(X_train, y_train)
            test_acc = knn.score(X_test, y_test)

            train_scores.append(train_acc)
            test_scores.append(test_acc)

        train_acc_arr.append(np.mean(train_scores))
        test_acc_arr.append(np.mean(test_scores))

    return test_sizes, train_acc_arr, test_acc_arr


def minkowskian_check(X, y, p_min, p_max, neighbors=5, test_size=0.2, rand_state=42):
    # test for changes in p on the same data set.
    p_range = p_max - p_min
    p_val_array = []
    train_acc_arr, test_acc_arr = [], []

    # since we want to test only the changes in p, we have a estatic data set defined here
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=rand_state
    )

    for i in range(p_range+1):
        # set up the classifier
        p_val = p_min + i
        p_val_array.append(p_val)
        knn = KNeighborsClassifier(n_neighbors=neighbors, p=p_val)
        knn.fit(X_train, y_train)

        # save the accuracies
        train_acc = knn.score(X_train, y_train)
        test_acc = knn.score(X_test, y_test)

        train_acc_arr.append(train_acc)
        test_acc_arr.append(test_acc)

    return p_val_array, train_acc_arr, test_acc_arr


def print_probs(n_rows, X, y, k=5, t=0.2, rand_state=42, p=2):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=t, random_state=rand_state
    )
    knn = KNeighborsClassifier(n_neighbors=k, p=p).fit(X_train, y_train)
    probs = knn.predict_proba(X_test)
    df = pd.DataFrame(probs, columns=knn.classes_, index=X_test.index).round(2)
    df["Predicted"] = knn.predict(X_test)
    df["Actual"] = y_test
    df["Correct"] = df["Predicted"] == df["Actual"]
    print(df.head(n_rows))


## READABILITY FUNCTIONS ##
def avg_array(arr):
    # take in an array and get the average
    return sum(arr) / len(arr)


# def mode_array(arr):
#     # take in an array and return the mode
#     return stats.mode(arr, keepdims=False)


def median_array(arr):
    # take in an array and return the median
    return np.median(arr)


def main():
    # Get the data
    url = "https://raw.githubusercontent.com/BYU-CS-472/CS472/master/datasets/glass_train.arff"
    with urllib.request.urlopen(url) as response:
        raw_data = response.read().decode("utf-8")

    # Put data into a dataframe
    data, _ = arff.loadarff(io.StringIO(raw_data))
    df = pd.DataFrame(data)

    # Define X and Y
    # X = the type column
    X = df.drop(columns=["Type"])
    # y = everything bar X
    y = df["Type"].str.decode("utf-8")

    # test/training splits test
    t_sizes, t_size_train_acc, t_size_test_acc = training_size_check(
        X, y, 0.1, 0.5, 0.05
    )
    print("Different Size Train/Test split averages")
    print(f"Training set\n{t_size_train_acc}\nTesting set\n{t_size_test_acc}")
    print(
        f"Train - Test\nOverall Avg Accuracy:\t{avg_array(t_size_train_acc)}\t{avg_array(t_size_test_acc)}"
    )
    # print(f"Mode accuracy:\t{mode_array(t_size_train_acc)}\t{mode_array(t_size_test_acc)}")
    print(
        f"Median accuracy:\t{median_array(t_size_train_acc)}\t{median_array(t_size_test_acc)}"
    )

    pd.DataFrame(
        {"Train": t_size_train_acc, "Test": t_size_test_acc},
        index=t_sizes,
    ).plot(
        marker="o",
        xlabel="Test Set Size (fraction of data)",
        ylabel="Mean Accuracy (10 varying random state splits)",
        title="KNN Accuracy vs Test Set Size",
        grid=True,
    )
    plt.show()
    print("Training/test size chart done. Moving to minkowskian split.")

    p_axis, p_train_acc, p_test_acc = minkowskian_check(X, y, 1, 10)
    print("Minkowskian Test Results:\nArray of Accuracy:")
    print("Training acc:\n", p_train_acc, "\nTesting acc:\n", p_test_acc)
    print(
        f"Train-Test\nAccuracy Avg:\t{avg_array(p_train_acc)}\t{avg_array(p_test_acc)}"
    )
    # print(f"Mode Acc:\t{mode_array(p_train_acc)}\t{mode_array(p_test_acc)}")
    print(f"Median Acc:\t{median_array(p_train_acc)}\t{p_test_acc}")

    pd.DataFrame(
        {"Train": p_train_acc, "Test": p_test_acc},
        index=p_axis,
    ).plot(
        marker="o",
        xlabel="p (Minkowskian Exponent)",
        ylabel="Accuracy",
        title="KNN Accuracy vs p (Minkowskian Exponent)",
        grid=True,
    )
    plt.show()

    print_probs(50, X, y)
    print("Done!")

if __name__ == "__main__":
    main()
