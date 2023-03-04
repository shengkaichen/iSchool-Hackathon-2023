from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def numericalize(df):
    """
    :param df: a dataframe
    :return: a dataframe with no object type
    """
    result = df.copy()
    key = {}
    le = preprocessing.LabelEncoder()
    for col in result:
        if result.dtypes[col] == np.object0 or result[col].dtype.name == "category":
            le.fit(result[col].astype(str))
            result[col] = le.transform(result[col].astype(str))
            key[col] = le.inverse_transform(np.unique(result[col]))
    return result, key


def normalization(df, scaler):
    result = df.copy()
    if scaler == "z-score":
        return pd.DataFrame(StandardScaler().fit_transform(result))
    if scaler == "min-max":
        return pd.DataFrame(MinMaxScaler().fit_transform(result))


def pca(d, df):
    model = PCA(n_components=d)
    model.fit(df)
    data_pca = model.transform(df)
    return pd.DataFrame(data_pca)


def read_label(df, col):
    """
    :param df: 1D dataframe that contains string label
    :param col: a string that contains name of the label column
    :return: entire dataframe with int label
    """
    df[col] = preprocessing.LabelEncoder().fit_transform(df[col])


def update_label(df, min_value, max_value):
    data = df.copy()
    data[df == 1] = min_value  # normal
    data[df == -1] = max_value  # outlier
    return data


def graph_frequency_numeric(df):
    """
    :param df: a dataframe only contains numeric value
    :return: a combine bar chart
    """
    row = len(df.columns)
    fig, axes = plt.subplots(figsize=(18, 36))
    for col in range(row):
        ax = plt.subplot(row, 2, col + 1)
        df[df.columns[col]].hist()
        ax.set_title(df.columns[col] + " (log)")
        ax.set_yscale('log')
    fig.tight_layout()
    plt.show()


def graph_frequency_string(df):
    """
    :param df: dataframe only contains object value
    :return: a bar chart
    """
    row = len(df.columns)
    fig, axes = plt.subplots(figsize=(27, 15))
    # Set general font size
    plt.rcParams["font.size"] = "12"
    for col in range(row):
        # get a series with the value count in each column
        count = df[df.columns[col]].value_counts()
        # transform series into dataframe
        data = count.reset_index(name="total").sort_values(by=["total"], ascending=True)
        # the label locations
        index = range(len(data[data.columns[0]]))
        ax = plt.subplot(row, 2, col + 1)
        ax.set_title(df.columns[col], fontsize=18)
        ax.set_ylabel("Frequency", labelpad=9, fontsize=12)
        # if the size of features is more than 5, only show the top 5 results on the chart.
        if index[-1] > 10:
            index = range(10)
            data = data.tail(10)
            ax.set_yticks(index, list(data[data.columns[0]]), rotation=0)
            ax.bar_label(plt.barh(index, data[data.columns[1]], align="edge"), fontsize=12, padding=6)
        else:
            ax.set_yticks(index, list(data[data.columns[0]]), rotation=0)
            ax.bar_label(plt.barh(index, data[data.columns[1]], align="edge"), fontsize=12, padding=6)
    fig.tight_layout()
    plt.show()


