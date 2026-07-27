import numpy as np
import time
class PerceptronMultiLayer:
    def __init__(self, eta=0.1, epochs=150, hidden_layer_sizes=(10,)):
        self.eta = eta
        self.epochs = epochs
        self.hidden_layer_sizes = hidden_layer_sizes
        self.weights = []
    
    def activation(self, u):
        return 1.0 / (1.0 + np.exp(-u))

    def activation_derivative(self, y):
        return y * (1.0 - y)
    
    def initialize_weights(self, n_features):
        layer_sizes = [n_features] + list(self.hidden_layer_sizes) + [1]
        self.weights = []
        for i in range(len(layer_sizes) - 1):
            w = np.random.rand(layer_sizes[i] + 1, layer_sizes[i + 1]) * 0.01
            self.weights.append(w)
    
    def fit(self, X, d):
        start_time = time.time()

        N, n_features = X.shape
        self.initialize_weights(n_features)

        d_01 = np.where(d == 1, 1, 0)

        for epoch in range(self.epochs):
            for i in range(N):
                x_i = X[i]
                d_i = d_01[i]

                activations = [x_i]
                current_input = x_i

                for l in range(len(self.weights)):
                    input_with_bias = np.hstack([1, current_input])
                    u = np.dot(input_with_bias, self.weights[l])
                    y = self.activation(u)
                    activations.append(y)
                    current_input = y
                
                y_pred = activations[-1]
                error = d_i - y_pred
                deltas = [error * self.activation_derivative(y_pred)]

                for l in range(len(self.weights) - 1, 0, -1):
                    w_no_bias = self.weights[l][1:, :]
                    hidden_delta = np.dot(deltas[0], w_no_bias.T) * self.activation_derivative(activations[l])
                    deltas.insert(0, hidden_delta)
                
                for l in range(len(self.weights)):
                    input_with_bias = np.hstack([1, activations[l]])
                    self.weights[l] += self.eta * np.outer(input_with_bias, deltas[l])
        elapsed_time = time.time() - start_time
        return elapsed_time
    
    def predict(self, X):
        N = X.shape[0]
        y_preds = []
        
        for i in range(N):
            current_input = X[i]
            
            # Forward pass para a amostra i
            for l in range(len(self.weights)):
                input_with_bias = np.hstack([1.0, current_input])
                u = np.dot(input_with_bias, self.weights[l])
                y = self.activation(u)
                current_input = y # A saída da camada vira entrada da próxima
                
            # A saída final da última camada
            y_pred_prob = current_input
            y_preds.append(y_pred_prob)
            
        y_preds = np.array(y_preds).flatten()
        
        # Converte as probabilidades de volta para as classes originais {-1, 1}
        y_pred_class = np.where(y_preds >= 0.5, 1, -1)
        
        return y_pred_class
