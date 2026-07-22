import pandas as pd
import numpy as np

def load_breast_cancer_data(filepath = "data/breast_cancer.csv"):
    df = pd.read_csv(filepath, header=None)
    y_raw = df.iloc[:, 1].values    
    d = np.where(y_raw == 'M', 1, -1)
    X = df.iloc[:, 2:].values
    return X, d
