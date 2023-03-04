import pandas as pd
from sklearn.model_selection import train_test_split
from models import AnomalyDetection
from helper_functions import Tools

df = pd.read_csv("./data2/creditcard.csv", sep=",", header=0)
Tools.read_label(df, "Class")

a1 = AnomalyDetection.isolation_forest(df, 0.2)
a2 = AnomalyDetection.elliptical_envelope(df, 0.2)
