    import pandas as pd
    from sklearn.model_selection import train_test_split
    from AnomalyDetection import AnomalyDetection
    df = pd.read_csv("./data/creditcard.csv", sep=",", header=0)
    AnomalyDetection.read_label(df, "Class")

    a1 = AnomalyDetection.isolation_forest(df, 0.2)
    a2 = AnomalyDetection.elliptical_envelope(df, 0.2)
