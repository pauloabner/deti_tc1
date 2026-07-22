
"""
This module contains mathematical utility functions.
"""
import numpy as np
import matplotlib.pyplot as plt

def add_bias(x, bias = 1):
    return [bias] + x

def dot_product(w, x):    
    if len(w) != len(x): raise ValueError("Vectors must be of the same length")
    return sum(a * b for a, b in zip(w, x))

def plot_decision_boundary(w, b, title="Decision Boundary"):
    """
    Plot the decision boundary for a linear classifier in 2D.
    
    Parameters:
    w (list): Weights of the classifier.
    b (float): Bias of the classifier.
    title (str): Title for the plot.
    """
    x = np.linspace(-5, 5, 100)
    y = -(w[0] * x + b) / w[1] if w[1] != 0 else np.zeros_like(x)
    
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label='Decision Boundary')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def matrix_X(dim, x):
    x = np.array(x)
    x_scaled = x / np.max(x)
    X = np.empty((x_scaled.shape[0], 0))
    
    for i in range(1, dim+2):
        # X=[X v.^(i-1)]
        column = (x_scaled ** (i - 1)).reshape(-1, 1)
        X = np.hstack((X, column))
    return X

def beta_estimate(X, y):
    # beta = (X^T X)^(-1) X^T y
    return np.linalg.pinv(X.T @ X) @ X.T @ y

def y_predict(X, beta):
    return X @ beta 

def sum_squared_error(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2)

def sum_mean_squared_error(y_true, y_pred):
    y_mean = np.mean(y_true)
    return np.sum((y_true - y_mean) ** 2)

def r_squared(y_true, y_pred):
    ss_res = sum_squared_error(y_true, y_pred)
    ss_tot = sum_mean_squared_error(y_true, y_true)
    return 1 - (ss_res / ss_tot)

class Gauss:
    def __init__(self, y, x, dim):
        self.beta = None
        self.y = y
        self.x = x
        self.dim = dim
        self.fit()


    def fit(self):
        self.X = matrix_X(self.dim, self.x)
        self.beta = beta_estimate(self.X, self.y)
        self.y_pred = y_predict(self.X, self.beta)
        self.r2 = r_squared(self.y, self.y_pred)
    
    def X(self):
        return self.X
    
    def y_pred(self):
        return self.y_pred
    
    def r2(self):
        return self.r2
    
    def beta(self):
        return self.beta
    
    def dim(self):
        return self.dim