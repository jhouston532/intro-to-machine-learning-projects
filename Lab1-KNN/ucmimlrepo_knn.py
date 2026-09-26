import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


# the metric, as given in the assignment details
def mixed_metric(x, y, NOMINAL_MASK):
    d = 0.0
    for i, is_nominal in enumerate(NOMINAL_MASK):
        if is_nominal:
            d += 0 if x[i] == y[i] else 1
        else:
            d += (x[i] - y[i]) ** 2
    return d**0.5


def main():
    # good golly this was a pain in the butt
    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/lymphography/lymphography.data"

    COLUMNS = [
        "class",
        "lymphatics",
        "block of affere",
        "bl. of lymph. c",
        "bl. of lymph. s",
        "by pass",
        "extravasates",
        "regeneration of",
        "early uptake in",
        "lym.nodes dimin",
        "lym.nodes enlar",
        "changes in lym",
        "defect in node",
        "changes in node",
        "changes in stru",
        "special forms",
        "dislocation of",
        "exclusion of no",
        "no. of nodes in",
    ]

    df = pd.read_csv(URL, header=None, names=COLUMNS)
    X = df.drop(columns=["class"])
    y = df["class"]

    # print(X.shape)
    # print(X.isna().sum())
    # print(X.head())

    CONTINUOUS = {"lym.nodes dimin", "lym.nodes enlar", "no. of nodes in"}
    NOMINAL_MASK = [name not in CONTINUOUS for name in X.columns]

    # use label encoding on X
    for i, is_nominal in enumerate(NOMINAL_MASK):
        if not is_nominal:
            continue

        column = X.iloc[:, i]
        encoder = LabelEncoder()
        encoded_column = encoder.fit_transform(column)
        X.iloc[:, i] = np.asarray(encoded_column)

    # transform into numpy for ease of use with metrics
    X_arr = X.to_numpy(dtype=float)
    y_arr = y.to_numpy()

    # define the testing set
    rand_state = 5
    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y_arr, test_size=0.2, random_state=rand_state
    )
    # do some normalization
    norm = MinMaxScaler()

    # get the indexes of the continuous variables
    cont_idx = [i for i, is_nominal in enumerate(NOMINAL_MASK) if not is_nominal]

    # apply the normalization to the numerical values
    X_train[:, cont_idx] = norm.fit_transform(X_train[:, cont_idx])
    X_test[:, cont_idx] = norm.transform(X_test[:, cont_idx])

    iter_num = 25
    train_scores, test_scores = [], []

    # let's try some variable Ks
    for i in range(iter_num):
        K = i + 1
        clf = KNeighborsClassifier(
            n_neighbors=K,
            weights="distance",
            metric=mixed_metric,
            metric_params={"NOMINAL_MASK": NOMINAL_MASK},
        )

        clf.fit(X_train, y_train)
        train_score = clf.score(X_train, y_train)
        test_score = clf.score(X_test, y_test)
        # print("Results")
        # print("\tK:\t\t", K)
        # print("\tTraining Set Score\t", train_score)
        # print("\tTesting Set Score\t", test_score, "\n\n")
        train_scores.append(train_score)
        test_scores.append(test_score)

    pd.DataFrame(
        {"Test Scores": test_scores, "Training Scores": train_scores},
        index=range(1, iter_num + 1),
    ).plot(
        marker="o",
        xlabel="K (# Neighbors)",
        ylabel="Accuracy Score",
        ylim=(0, 1.05),
        title=f"Accuracy vs K (# Neigbors), R={rand_state}",
        grid=True,
    )

    plt.show()


if __name__ == "__main__":
    main()
