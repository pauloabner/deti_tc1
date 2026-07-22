import numpy as np
from utils.dataset import load_breast_cancer_data
from utils.preprocessing import zscore_train_test
from models.lqm import LinearLeastSquaresClassifier

if __name__ == "__main__":
    ptrn = 0.8
    Nr = 50

    filepath = "data/wdbc.data"
    X, d = load_breast_cancer_data(filepath)
    N_total = X.shape[0]
    N_train = int(N_total * ptrn)

    for r in range(Nr):
        idx = np.random.permutation(N_total)
        train_idx = idx[:N_train]
        test_idx = idx[N_train:]

        X_train, X_test = X[train_idx], X[test_idx]
        d_train, d_test = d[train_idx], d[test_idx]

        X_train_norm, X_test_norm = zscore_train_test(X_train, X_test)

        model = LinearLeastSquaresClassifier()
        train_time = model.fit(X_train_norm, d_train)

        
