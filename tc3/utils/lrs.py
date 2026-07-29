import numpy as np
from utils.abner_math import predict_polynomial, calculate_sse
def local_random_search(x_data, y_data, lower_bound, upper_bound, iterations, dimension, std_dev):
    """Implementação do Local Random Search."""
    # 1. Solução inicial aleatória
    x_best = np.random.uniform(lower_bound, upper_bound, dimension)
    y_pred_best = predict_polynomial(x_data, x_best)
    f_best = calculate_sse(y_data, y_pred_best)
    
    error_history = [f_best]
    
    # 2. Loop de otimização
    for _ in range(iterations):
        # Ruído Gaussiano (Gaussian Noise)
        noise = np.random.normal(0, std_dev, dimension)
        x_cand = x_best + noise
        
        # Respeitar limites
        x_cand = np.clip(x_cand, lower_bound, upper_bound)
        
        # Avaliar candidato
        y_pred_cand = predict_polynomial(x_data, x_cand)
        f_cand = calculate_sse(y_data, y_pred_cand)
        
        # Minimizar erro
        if f_cand < f_best:
            x_best = x_cand
            f_best = f_cand
            
        error_history.append(f_best)
        
    return x_best, f_best, error_history
