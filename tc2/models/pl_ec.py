import numpy as np
import time

class PerceptronLogisticoEC:
    def __init__(self, eta=0.1, epochs=150):
        self.eta = eta
        self.epochs = epochs
        self.w = None

    def activation(self, u):
        denominator = 1.0 + np.exp(-u)
        denominator[denominator == 0] = 1e-10  # Evitar divisão por zero
        return 1/denominator

    def fit(self, X, d):
        start_time = time.time()

        N, num_attributes = X.shape
        X_bias = np.hstack([np.ones((N, 1)), X])
        np.random.seed(42)
        self.w = np.random.rand(num_attributes + 1) * 0.01

        d_01 = np.where(d == 1, 1, 0)

        for epoch in range(self.epochs):
            for i in range(N):
                x_i = X_bias[i]
                d_i = d_01[i]

                u = np.dot(self.w, x_i)
                y = self.activation(u)
                err = d_i - y
                self.w += self.eta * err * x_i
        
        elapsed_time = time.time() - start_time
        return elapsed_time

    def predict(self, X):
        N = X.shape[0]
        X_bias = np.hstack([np.ones((N, 1)), X])
        u = np.dot(X_bias, self.w)
        y = self.activation(u)
        y_pred = np.where(y >= 0.5, 1, -1)  
        return y_pred