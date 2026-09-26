import io
import urllib.request

import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import arff
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler


def knr_iteration(
    df: pd.DataFrame,
    r_state: int,
    input_norm: bool = False,
    dist_weight: bool = False,
    k_neighbors: int = 3,
    y_class: str = "MEDV",
):

    X = df.drop(columns=[y_class])
    y = df[y_class]

    # transform the X based on input_norm
    if input_norm:  # is true
        normalizer = MinMaxScaler()
        X = normalizer.fit_transform(X)

    # create and fit model as knr
    # params based on dist_weight
    if dist_weight:
        knr = KNeighborsRegressor(n_neighbors=k_neighbors, weights="distance")
    else:
        knr = KNeighborsRegressor(n_neighbors=k_neighbors, weights="uniform")

    # create the test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=r_state
    )

    # fit the model
    knr.fit(X_train, y_train)

    # capture the model's scores
    mae_results = []
    score_results = []

    mae_results.append(mean_absolute_error(y_train, knr.predict(X_train)))
    mae_results.append(mean_absolute_error(y_test, knr.predict(X_test)))

    score_results.append(knr.score(X_train, y_train))
    score_results.append(knr.score(X_test, y_test))

    # return score and MAE
    return mae_results  # , score_results


def main():
    url = "https://raw.githubusercontent.com/BYU-CS-472/CS472/master/datasets/housing_train.arff"

    # get the data from the provided link
    with urllib.request.urlopen(url) as response:
        raw_data = response.read().decode("utf-8")

    # Put data into a dataframe
    data, meta = arff.loadarff(io.StringIO(raw_data))
    df = pd.DataFrame(data)
    df.columns = meta.names()

    # df = df.drop(columns=[b'B']) # remove the offending data
    df["CHAS"] = df["CHAS"].astype(int)  # in case it decides to use bytestrings
    num_itr = 15  # change this variable to change how many data points are used.
    test_mae_to_k = []
    for i in range(num_itr):
        k = i + 1
        mae_arr = knr_iteration(df, 1, True, True, k)
        test_mae_to_k.append(mae_arr[1])

    pd.DataFrame({"MAE (Test Set)": test_mae_to_k}, index=range(1, num_itr + 1)).plot(
        marker="o",
        xlabel="K (# Neighbors)",
        ylabel="MAE (Test Set)",
        ylim=(0, 5),
        title="Test Set MAE vs K (Number of Neighbors)",
        grid=True,
    )
    plt.show()


if __name__ == "__main__":
    main()
