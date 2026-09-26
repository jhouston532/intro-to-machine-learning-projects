import io
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


def main(): 
    url = "https://raw.githubusercontent.com/BYU-CS-472/CS472/master/datasets/housing_train.arff"

    # get the data from the provided link
    with urllib.request.urlopen(url) as response:
        raw_data = response.read().decode("utf-8")

    # Put data into a dataframe
    data = arff.loadarff(io.StringIO(raw_data))
    df = pd.DataFrame(data)
    df = df.drop(columns=['B']) # remove the offending data 
    df['CHAS'] = df['CHAS'].astype(int) # in case it decides to use bytestrings 



