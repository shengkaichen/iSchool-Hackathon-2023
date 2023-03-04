"""
Dataset resource: https://datahub.io/machine-learning/creditcard#data
"""
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def isolation_forest(df, ratio):
    """
    Isolation Forest: A classification based algorithm (Decision Tree)
    resource: https://hands-on.cloud/using-python-and-isolation-forest-algorithm-for-anomalies-detection/
    :return: a dataframe with outlier data
    """
    # split to train and test data
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df.iloc[:, -1],
                                                        test_size=ratio, random_state=3)

    # train model
    model = IsolationForest(max_samples=len(X_train))
    model.fit(X_train)
    y_pred = model.predict(X_test)

    # change the predicted values to fit the dataset labels
    y_pred = update_label(y_pred, y_train.min(), y_train.max())
    X_test['anomalies'] = y_pred
    anomalies = X_test[X_test.anomalies == y_train.max()]

    # get the accuracy
    print(accuracy_score(y_pred, y_test))
    return anomalies


def elliptical_envelope(df, ratio):
    """
    Elliptical Envelope: A clustering based algorithm
    resource: https://towardsdatascience.com/machine-learning-for-anomaly-detection-elliptic-envelope-2c90528df0a6
    :return: a dataframe with outlier data
    """
    # split to train and test data
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df.iloc[:, -1],
                                                        test_size=ratio, random_state=3)

    # "contamination" parameter defines the proportion of values that will be identified as outliers with value ranges
    # between 0 and 0.5
    model = EllipticEnvelope(contamination=0.01)
    model.fit(X_train)
    y_pred = model.predict(X_test)

    # change the predicted values to fit the dataset labels
    y_pred = update_label(y_pred, y_train.min(), y_train.max())
    X_test['anomalies'] = y_pred
    anomalies = X_test[X_test.anomalies == y_train.max()]

    # get the accuracy
    print(accuracy_score(y_pred, y_test))
    return anomalies


















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
