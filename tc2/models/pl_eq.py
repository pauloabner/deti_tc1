import numpy as np
import time

def __init__(self, eta=0.1, epochs=150):
    self.eta = eta
    self.epochs = epochs
    self.w = None

def activation(self, u):
    return np.tanh(u)

def activation_derivative(self, u):
    return 1.0 - np.tanh(u) ** 2

def fit(self, X, d):
    start_time = time.time()

    N, num_attributes = X.shape
    X_bias = np.hstack([np.ones((N, 1)), X])
    np.random.seed(42)
    self.w = np.random.rand(num_attributes + 1) * 0.01

    for epoch in range(self.epochs):
        for i in range(N):
            x_i = X_bias[i]
            d_i = d[i]

            u = np.dot(self.w, x_i)
            y = self.activation(u)
            err = d_i - y
            self.w += self.eta * err * self.activation_derivative(u) * x_i
    
    elapsed_time = time.time() - start_time
    return elapsed_time

def predict(self, X):
    N = X.shape[0]
    X_bias = np.hstack([np.ones((N, 1)), X])
    u = np.dot(X_bias, self.w)
    y = self.activation(u)
    y_pred = np.sign(y)
    y_pred[y_pred == 0] = 1  # Atribuir +1 para valores de y iguais a zero
    return y_pred