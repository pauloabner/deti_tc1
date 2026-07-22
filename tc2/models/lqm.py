import numpy as np
import time
class LinearLeastSquaresClassifier:
    def __init__(self):
        self.w = None
    
    def fit(self, X, d):
        start_time = time.time()

        N = X.shape[0]
        X_bias = np.hstack([np.ones((N, 1)), X])

        # Solução analítica usando a pseudo-inversa de Moore-Penrose (np.linalg.pinv)
        # w = (X^T * X)^(-1) * X^T * d
        self.w = np.linalg.pinv(X_bias) @ d

        elapsed_time = time.time() - start_time
        return elapsed_time