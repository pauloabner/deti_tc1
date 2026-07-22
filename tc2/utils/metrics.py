import numpy as np
def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred) * 100.0

def compute_statistics(metrics_list, times_list):
    arr_metrics = np.array(metrics_list)
    arr_times = np.array(times_list)

    stats = {
        "media": np.mean(arr_metrics),
        "minimo": np.min(arr_metrics),
        "maximo": np.max(arr_metrics),
        "mediana": np.median(arr_metrics),
        "desvio_padrao": np.std(arr_metrics, ddof=1), # Desvio padrão amostral
        "tempo_total": np.sum(arr_times)
    }
    return stats