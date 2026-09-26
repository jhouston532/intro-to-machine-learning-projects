import io
import urllib.request

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
    y_class: str = "MEDV",
    k_neighbors: int = 3,
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
    return mae_results, score_results


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

    # nothing turned on iteration
    mae_arr1, score_arr1 = knr_iteration(df, 1)
    print(
        f"""\'Nothing On\' Set Results\nTraining Set Results\n\tMAE: {mae_arr1[0]}\n\tScore: {score_arr1[0]}\n\nTesting Set Results\n\tMAE: {mae_arr1[1]}\n\tScore: {score_arr1[1]}\n\n"""
    )

    mae_arr2, score_arr2 = knr_iteration(df, 1, True)
    print(
        f"""\'Normalization On\' Set Results\nTraining Set Results\n\tMAE: {mae_arr2[0]}\n\tScore: {score_arr2[0]}\n\nTesting Set Results\n\tMAE: {mae_arr2[1]}\n\tScore: {score_arr2[1]}\n\n"""
    )

    mae_arr3, score_arr3 = knr_iteration(df, 1, True, True)
    print(
        f"""\'Everything On\' Set Results\nTraining Set Results\n\tMAE: {mae_arr3[0]}\n\tScore: {score_arr3[0]}\n\nTesting Set Results\n\tMAE: {mae_arr3[1]}\n\tScore: {score_arr3[1]}\n\n"""
    )


if __name__ == "__main__":
    main()
