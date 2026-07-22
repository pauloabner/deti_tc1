import numpy as np
from utils.dataset import load_breast_cancer_data

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