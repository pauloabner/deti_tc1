import numpy as np

def zscore_train_test(X_train, X_test):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    std[std == 0.0] = 1.0
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std

    return X_train_norm, X_test_norm

def no_normalization(X_train, X_test):
    """Retorna os dados originais (Sem Normalização)"""
    return X_train, X_test

def minmax_01_normalize(X_train, X_test):
    """Normalização por mudança de escala para o intervalo [0, +1]"""
    min_val = np.min(X_train, axis=0)
    max_val = np.max(X_train, axis=0)
    range_val = max_val - min_val
    
    # Evita divisão por zero
    range_val[range_val == 0.0] = 1.0
    
    X_train_norm = (X_train - min_val) / range_val
    X_test_norm = (X_test - min_val) / range_val
    
    return X_train_norm, X_test_norm

def minmax_bipolar_normalize(X_train, X_test):
    """Normalização por mudança de escala para o intervalo [-1, +1]"""
    # Primeiro escala para [0, 1]
    X_train_01, X_test_01 = minmax_01_normalize(X_train, X_test)
    
    # Depois converte de [0, 1] para [-1, +1] usando: Y = X * 2 - 1
    X_train_norm = X_train_01 * 2.0 - 1.0
    X_test_norm = X_test_01 * 2.0 - 1.0
    
    return X_train_norm, X_test_norm

def apply_pca(X_train, X_test, num_componente):
    cov_matrix = np.cov(X_train, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    total_variance = np.sum(eigenvalues)
    explained_variance_ratio = eigenvalues / total_variance

    projection_matrix = eigenvectors[:, :num_componente]
    X_train_pca = np.dot(X_train, projection_matrix)
    X_test_pca = np.dot(X_test, projection_matrix)

    variance_retained = np.sum(explained_variance_ratio[:num_componente]) *100

    return X_train_pca, X_test_pca, variance_retained
